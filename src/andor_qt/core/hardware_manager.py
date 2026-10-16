"""Hardware manager for shared hardware instances.

This module provides a singleton HardwareManager that shares camera and
spectrograph instances between the GUI and PyMeasure procedures.
"""

from __future__ import annotations

import logging
import os
import sys
import threading
from typing import TYPE_CHECKING, Callable, Optional, Type

from PySide6.QtCore import QTimer

from andor_qt.core.event_bus import get_event_bus
from andor_qt.core.signals import get_hardware_signals

if TYPE_CHECKING:
    import numpy as np

    from andor_pymeasure.instruments.andor_camera import AndorCamera
    from andor_pymeasure.instruments.andor_spectrograph import AndorSpectrograph
    from andor_pymeasure.instruments.thorlabs_mc2000b import ChopperPair
    from pymeasure.experiment import Procedure

    from andor_qt.core.config import AppConfig
    from andor_qt.core.motion_manager import MotionControllerManager

log = logging.getLogger(__name__)


class HardwareManager:
    """Singleton managing shared hardware instances.

    This class ensures that camera and spectrograph are initialized once
    and shared between the GUI controls and PyMeasure procedures.

    Usage:
        manager = HardwareManager.instance()
        manager.initialize()

        # Inject into procedure class
        manager.inject_into_procedure(SpectrumProcedure)

        # Direct access
        temp = manager.camera.temperature
    """

    _instance: Optional["HardwareManager"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "HardwareManager":
        """Ensure singleton instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    @classmethod
    def instance(cls) -> "HardwareManager":
        """Get the singleton instance."""
        return cls()

    def __init__(self):
        """Initialize the hardware manager."""
        if self._initialized:
            return

        self._camera: Optional["AndorCamera"] = None
        self._spectrograph: Optional["AndorSpectrograph"] = None
        self._motion_manager: Optional["MotionControllerManager"] = None
        self._chopper_pair: Optional["ChopperPair"] = None
        self._mock_mode = os.environ.get("ANDOR_MOCK", "0") == "1"
        self._signals = get_hardware_signals()
        self._temp_timer: Optional[QTimer] = None
        self._shutdown_in_progress = False
        self._hardware_lock = threading.Lock()
        self._sdk_path: str = r"C:\Program Files\Andor SDK"
        self._config: Optional["AppConfig"] = None
        self._initialized = True

    @property
    def _event_bus(self):
        """Get the current EventBus instance (dynamic for test isolation)."""
        return get_event_bus()

    @property
    def camera(self) -> Optional["AndorCamera"]:
        """Get the camera instance."""
        return self._camera

    @property
    def spectrograph(self) -> Optional["AndorSpectrograph"]:
        """Get the spectrograph instance."""
        return self._spectrograph

    @property
    def motion_manager(self) -> Optional["MotionControllerManager"]:
        """Get the motion controller manager."""
        return self._motion_manager

    @property
    def chopper_pair(self) -> Optional["ChopperPair"]:
        """Get the MC2000B chopper pair (pump master + probe slave)."""
        return self._chopper_pair

    @property
    def is_initialized(self) -> bool:
        """Check if hardware is initialized."""
        return self._camera is not None and self._spectrograph is not None

    @property
    def mock_mode(self) -> bool:
        """Check if running in mock mode."""
        return self._mock_mode

    @property
    def sdk_path(self) -> str:
        """Get the SDK path."""
        return self._sdk_path

    def set_config(self, config: "AppConfig") -> None:
        """Set configuration from AppConfig.

        Args:
            config: Application configuration.
        """
        self._config = config
        self._sdk_path = config.hardware.sdk_path
        self._mock_mode = config.hardware.mock_mode

    def initialize(
        self,
        sdk_path: str = r"C:\Program Files\Andor SDK",
        on_complete: Optional[Callable] = None,
        on_error: Optional[Callable[[str], None]] = None,
    ) -> None:
        """Initialize camera and spectrograph in a background thread.

        Args:
            sdk_path: Path to the Andor SDK installation.
            on_complete: Callback when initialization completes.
            on_error: Callback with error message if initialization fails.
        """
        if self.is_initialized:
            log.warning("Hardware already initialized")
            if on_complete:
                on_complete()
            return

        def _init_thread():
            try:
                self._do_initialize(sdk_path)
                if on_complete:
                    on_complete()
            except Exception as e:
                log.error(f"Hardware initialization failed: {e}")
                if on_error:
                    on_error(str(e))
                self._signals.error_occurred.emit("HardwareManager", str(e))

        thread = threading.Thread(target=_init_thread, daemon=True)
        thread.start()

    def _do_initialize(self, sdk_path: str) -> None:
        """Perform hardware initialization (called in background thread)."""
        if self._mock_mode:
            log.info("Initializing in MOCK mode")
            self._init_mock_hardware()
        else:
            log.info("Initializing real hardware")
            self._init_real_hardware(sdk_path)

        # Emit signals
        if self._camera:
            self._signals.camera_initialized.emit({
                "xpixels": self._camera.xpixels,
                "ypixels": self._camera.ypixels,
                "serial": self._camera.info.serial_number if self._camera.info else "Unknown",
            })

        if self._spectrograph:
            self._signals.spectrograph_initialized.emit({
                "serial": self._spectrograph.info.serial_number if self._spectrograph.info else "Unknown",
                "num_gratings": self._spectrograph.info.num_gratings if self._spectrograph.info else 0,
            })
            # Emit initial wavelength
            self._signals.wavelength_changed.emit(self._spectrograph.wavelength)

        # Publish to EventBus
        self._event_bus.publish(
            "hardware.initialized",
            camera={
                "xpixels": self._camera.xpixels if self._camera else 0,
                "ypixels": self._camera.ypixels if self._camera else 0,
            },
            spectrograph={
                "num_gratings": self._spectrograph.info.num_gratings if self._spectrograph and self._spectrograph.info else 0,
            },
        )

        log.info("Hardware initialization complete")

    def _init_mock_hardware(self) -> None:
        """Initialize mock hardware for development."""
        # Patch SDK imports with mocks
        from andor_pymeasure.instruments.mock import create_mock_sdk_modules

        mock_modules = create_mock_sdk_modules()
        sys.modules.update(mock_modules)

        from andor_pymeasure.instruments.andor_camera import AndorCamera
        from andor_pymeasure.instruments.andor_spectrograph import AndorSpectrograph

        self._camera = AndorCamera(sdk_path="mock")
        self._camera.initialize()

        self._spectrograph = AndorSpectrograph(sdk_path="mock")
        self._spectrograph.initialize()

        # Initialize motion controllers
        self._init_motion_controllers()

        # Initialize chopper pair (mock objects, never opens a serial port)
        self._init_chopper_pair()

    def _init_real_hardware(self, sdk_path: str) -> None:
        """Initialize real hardware."""
        from andor_pymeasure.instruments.andor_camera import AndorCamera
        from andor_pymeasure.instruments.andor_spectrograph import AndorSpectrograph

        self._camera = AndorCamera(sdk_path=sdk_path)
        self._camera.initialize()

        self._spectrograph = AndorSpectrograph(sdk_path=sdk_path)
        self._spectrograph.initialize()

        # Initialize motion controllers
        self._init_motion_controllers()

        # Initialize chopper pair (best-effort: a missing port is logged but
        # does not abort hardware init)
        self._init_chopper_pair()

    def _init_motion_controllers(self) -> None:
        """Initialize motion controllers from config."""
        from andor_qt.core.motion_manager import MotionControllerManager

        # Get motion config from AppConfig if available
        if self._config and hasattr(self._config, "motion_controllers"):
            motion_config = self._config.motion_controllers
        else:
            # Default mock config for development
            motion_config = {
                "enabled": True,
                "controllers": [
                    {
                        "name": "mock_stage",
                        "type": "mock",
                        "home_on_startup": False,
                        "axes": [
                            {
                                "name": "delay",
                                "index": 1,
                                "position_min": 0.0,
                                "position_max": 300.0,
                                "velocity": 10.0,
                                "units": "ps",
                            }
                        ],
                    }
                ],
            }

        self._motion_manager = MotionControllerManager(motion_config)
        self._motion_manager.initialize()

        # Emit motion initialized signal
        if self._motion_manager.all_axes:
            axis_info = {
                name: {
                    "position": axis.position,
                    "position_min": axis.position_min,
                    "position_max": axis.position_max,
                    "units": axis.units,
                }
                for name, axis in self._motion_manager.all_axes.items()
            }
            self._signals.motion_initialized.emit(axis_info)
            log.info(f"Motion controllers initialized: {list(axis_info.keys())}")

    def _init_chopper_pair(self) -> None:
        """Initialize the MC2000B chopper pair (pump master + probe slave).

        In mock mode, two ``MockMC2000B`` objects are created with no IO. In
        hardware mode, each port is opened in a try block — a failure on
        either port logs a warning and that slot stays None, but does NOT
        abort hardware init.
        """
        chopper_cfg = getattr(self._config, "choppers", None) if self._config else None
        if chopper_cfg is not None and not getattr(chopper_cfg, "enabled", True):
            log.info("Chopper pair disabled via config")
            self._chopper_pair = None
            self._signals.choppers_initialized.emit(
                {"pump_port": "", "probe_port": "",
                 "pump_connected": False, "probe_connected": False}
            )
            return

        from andor_pymeasure.instruments.thorlabs_mc2000b import (
            ChopperPair,
            MC2000B,
            MockMC2000B,
        )

        pump_port = getattr(chopper_cfg, "pump_port", "COM4") if chopper_cfg else "COM4"
        probe_port = getattr(chopper_cfg, "probe_port", "COM3") if chopper_cfg else "COM3"

        if self._mock_mode:
            pump = MockMC2000B(port=pump_port)
            probe = MockMC2000B(port=probe_port) if probe_port else None
            self._chopper_pair = ChopperPair(pump=pump, probe=probe)
            log.info("Mock chopper pair created (pump=%s, probe=%s)",
                     pump_port, probe_port or "<none>")
            self._signals.choppers_initialized.emit(
                {"pump_port": pump_port, "probe_port": probe_port or "",
                 "pump_connected": True, "probe_connected": probe is not None}
            )
            return

        pump = None
        probe = None
        if pump_port:
            try:
                pump = MC2000B(pump_port)
                log.info("Pump chopper opened on %s: %s", pump_port, pump.id)
            except Exception as e:
                log.warning("Could not open pump chopper on %s: %s", pump_port, e)
                pump = None
        if probe_port:
            try:
                probe = MC2000B(probe_port)
                log.info("Probe chopper opened on %s: %s", probe_port, probe.id)
            except Exception as e:
                log.warning("Could not open probe chopper on %s: %s", probe_port, e)
                probe = None
        if pump is None and probe is None:
            self._chopper_pair = None
        else:
            self._chopper_pair = ChopperPair(pump=pump, probe=probe)
        self._signals.choppers_initialized.emit(
            {"pump_port": pump_port, "probe_port": probe_port,
             "pump_connected": pump is not None,
             "probe_connected": probe is not None}
        )

    def inject_into_procedure(self, procedure_class: Type["Procedure"]) -> None:
        """Inject shared hardware references into a procedure class.

        This allows procedures to use the shared hardware instead of
        initializing their own instances.

        Args:
            procedure_class: The procedure class to inject hardware into.
        """
        procedure_class._shared_camera = self._camera
        procedure_class._shared_spectrograph = self._spectrograph
        procedure_class._shared_motion_manager = self._motion_manager

    def set_axis_position(
        self,
        axis_name: str,
        position: float,
        units: str = "ps",
        on_complete: Optional[Callable] = None,
    ) -> None:
        """Set position of a named motion axis (background operation).

        Args:
            axis_name: Name of the axis to move.
            position: Target position value.
            units: Position units ("ps", "mm", "deg"). Default is "ps".
            on_complete: Callback when move completes.
        """
        if self._motion_manager is None:
            log.error("Motion manager not initialized")
            return

        axis = self._motion_manager.get_axis(axis_name)
        if axis is None:
            log.error(f"Axis '{axis_name}' not found")
            return

        self._signals.axis_position_changing.emit(axis_name, position)
        self._signals.axis_moving.emit(axis_name, True)

        def _move_thread():
            try:
                with self._hardware_lock:
                    if units == "ps":
                        axis.position_ps = position
                    elif units == "mm":
                        axis.position = position
                    else:
                        # For other units, assume mm
                        axis.position = position

                self._signals.axis_position_changed.emit(axis_name, axis.position)
                self._signals.axis_moving.emit(axis_name, False)
                self._event_bus.publish(
                    "hardware.axis_position_changed",
                    axis_name=axis_name,
                    position=axis.position,
                )
                if on_complete:
                    on_complete()
            except Exception as e:
                log.error(f"Error setting axis position: {e}")
                self._signals.error_occurred.emit("AxisMove", str(e))
                self._signals.axis_moving.emit(axis_name, False)

        thread = threading.Thread(target=_move_thread, daemon=True)
        thread.start()

    def start_temperature_polling(self, interval_ms: int = 2000) -> None:
        """Start polling camera temperature at regular intervals.

        Args:
            interval_ms: Polling interval in milliseconds.
        """
        if self._temp_timer is not None:
            return

        self._temp_timer = QTimer()
        self._temp_timer.timeout.connect(self._poll_temperature)
        self._temp_timer.start(interval_ms)
        log.debug(f"Temperature polling started (interval: {interval_ms}ms)")

    def stop_temperature_polling(self) -> None:
        """Stop temperature polling."""
        if self._temp_timer is not None:
            self._temp_timer.stop()
            self._temp_timer = None
            log.debug("Temperature polling stopped")

    def _poll_temperature(self) -> None:
        """Poll current temperature and emit signal."""
        if self._camera is None or self._shutdown_in_progress:
            return

        try:
            temp = self._camera.temperature
            status = self._camera.temperature_status
            self._signals.temperature_changed.emit(temp, status)
        except Exception as e:
            log.error(f"Error polling temperature: {e}")

    def set_cooler(
        self,
        on: bool,
        target_temp: int = -60,
        on_complete: Optional[Callable] = None,
    ) -> None:
        """Set cooler state (immediate action, not queued).

        Args:
            on: True to turn cooler on, False to turn off.
            target_temp: Target temperature in Celsius.
            on_complete: Callback when operation completes.
        """
        if self._camera is None:
            log.error("Camera not initialized")
            return

        def _cooler_thread():
            try:
                with self._hardware_lock:
                    if on:
                        self._camera.cooler_on(target=target_temp)
                    else:
                        self._camera.cooler_off()

                self._signals.cooler_state_changed.emit(on, target_temp)
                if on_complete:
                    on_complete()
            except Exception as e:
                log.error(f"Error setting cooler: {e}")
                self._signals.error_occurred.emit("Cooler", str(e))

        thread = threading.Thread(target=_cooler_thread, daemon=True)
        thread.start()

    def set_grating(
        self,
        grating: int,
        on_complete: Optional[Callable] = None,
    ) -> None:
        """Set grating (background operation).

        Args:
            grating: Grating index (1-indexed).
            on_complete: Callback when operation completes.
        """
        if self._spectrograph is None:
            log.error("Spectrograph not initialized")
            return

        self._signals.grating_changing.emit(grating)

        def _grating_thread():
            try:
                with self._hardware_lock:
                    self._spectrograph.grating = grating

                self._signals.grating_changed.emit(grating)
                self._event_bus.publish("hardware.grating_changed", grating=grating)
                if on_complete:
                    on_complete()
            except Exception as e:
                log.error(f"Error setting grating: {e}")
                self._signals.error_occurred.emit("Grating", str(e))

        thread = threading.Thread(target=_grating_thread, daemon=True)
        thread.start()

    def set_wavelength(
        self,
        wavelength: float,
        on_complete: Optional[Callable] = None,
    ) -> None:
        """Set wavelength (background operation).

        Args:
            wavelength: Center wavelength in nm.
            on_complete: Callback when operation completes.
        """
        if self._spectrograph is None:
            log.error("Spectrograph not initialized")
            return

        self._signals.wavelength_changing.emit(wavelength)

        def _wavelength_thread():
            try:
                with self._hardware_lock:
                    self._spectrograph.wavelength = wavelength

                self._signals.wavelength_changed.emit(wavelength)
                self._event_bus.publish("hardware.wavelength_changed", wavelength=wavelength)
                if on_complete:
                    on_complete()
            except Exception as e:
                log.error(f"Error setting wavelength: {e}")
                self._signals.error_occurred.emit("Wavelength", str(e))

        thread = threading.Thread(target=_wavelength_thread, daemon=True)
        thread.start()

    def get_wavelengths(self, hbin: int = 1) -> "np.ndarray":
        """Get wavelength array for current spectrograph/camera settings.

        Returns:
            Wavelength array in nm, or empty array if not available.
        """
        import numpy as np
        cal = self.get_calibration(hbin=hbin)
        return cal if cal is not None else np.array([])

    def get_calibration(self, hbin: int = 1) -> Optional["np.ndarray"]:
        """Get wavelength calibration for current settings.

        Args:
            hbin: Horizontal binning factor. Calibration array will have
                  xpixels // hbin points.

        Returns:
            Wavelength array or None if not available.
        """
        if self._spectrograph is None or self._camera is None:
            return None

        try:
            pixel_width = self._camera.info.pixel_width if self._camera.info else 26.0
            with self._hardware_lock:
                calibration = self._spectrograph.get_binned_calibration(
                    self._camera.xpixels, pixel_width, hbin
                )
            self._signals.calibration_updated.emit(calibration)
            return calibration
        except Exception as e:
            log.error(f"Error getting calibration: {e}")
            return None

    def shutdown(
        self,
        warmup: bool = True,
        on_complete: Optional[Callable] = None,
        on_progress: Optional[Callable[[str], None]] = None,
    ) -> None:
        """Shutdown hardware safely in background thread.

        Args:
            warmup: Whether to warm up camera before shutdown.
            on_complete: Callback when shutdown completes.
            on_progress: Callback with progress messages.
        """
        if not self.is_initialized:
            log.warning("Hardware not initialized, nothing to shutdown")
            if on_complete:
                on_complete()
            return

        self._shutdown_in_progress = True
        self.stop_temperature_polling()

        def _shutdown_thread():
            try:
                if on_progress:
                    on_progress("Stopping any running acquisitions...")

                # Abort any running acquisition
                if self._camera:
                    try:
                        self._camera.abort_acquisition()
                    except Exception:
                        pass

                # Warm up camera if needed
                if warmup and self._camera:
                    temp = self._camera.temperature
                    if temp < -20:
                        if on_progress:
                            on_progress(f"Warming up camera ({temp:.1f}°C → -20°C)...")

                        # Turn off cooler to allow warming
                        self._camera.cooler_off()

                        # Monitor warmup with progress updates
                        import time
                        start_time = time.time()
                        timeout = 300
                        while (time.time() - start_time) < timeout:
                            temp = self._camera.temperature
                            status = self._camera.temperature_status
                            if on_progress:
                                on_progress(f"Warming up camera: {temp:.1f}°C → -20°C")
                            # Emit temperature signal so UI can show progress
                            self._signals.temperature_changed.emit(temp, status)
                            if temp >= -20:
                                break
                            time.sleep(2)  # Check every 2 seconds

                # Shutdown camera
                if self._camera:
                    if on_progress:
                        on_progress("Shutting down camera...")
                    self._camera.shutdown()
                    # Don't emit signal from background thread

                # Shutdown spectrograph
                if self._spectrograph:
                    if on_progress:
                        on_progress("Shutting down spectrograph...")
                    self._spectrograph.shutdown()
                    # Don't emit signal from background thread

                # Shutdown motion controllers
                if self._motion_manager:
                    if on_progress:
                        on_progress("Shutting down motion controllers...")
                    self._motion_manager.shutdown()

                # Shutdown chopper pair (disables and closes serial ports)
                if self._chopper_pair:
                    if on_progress:
                        on_progress("Disabling choppers...")
                    try:
                        self._chopper_pair.shutdown()
                    except Exception as e:
                        log.warning("Chopper shutdown raised: %s", e)

                self._camera = None
                self._spectrograph = None
                self._motion_manager = None
                self._chopper_pair = None
                self._shutdown_in_progress = False

                # Publish to EventBus
                self._event_bus.publish("hardware.shutdown")

                log.info("Hardware shutdown complete")
                if on_complete:
                    on_complete()

            except Exception as e:
                log.error(f"Error during hardware shutdown: {e}")
                self._signals.error_occurred.emit("Shutdown", str(e))
                self._shutdown_in_progress = False

        thread = threading.Thread(target=_shutdown_thread, daemon=True)
        thread.start()

    @classmethod
    def reset_instance(cls) -> None:
        """Reset the singleton instance (for testing)."""
        with cls._lock:
            if cls._instance is not None:
                cls._instance._camera = None
                cls._instance._spectrograph = None
                cls._instance._motion_manager = None
                cls._instance._chopper_pair = None
                cls._instance._initialized = False
            cls._instance = None
