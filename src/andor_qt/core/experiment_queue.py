"""Experiment queue runner for sequential procedure execution.

Runs queued procedures sequentially in a background thread,
emitting Qt signals for progress and data.
"""

from __future__ import annotations

import logging
import threading
from collections import deque
from typing import TYPE_CHECKING, Deque, Optional, Tuple

import numpy as np
from PySide6.QtCore import QObject, Signal

if TYPE_CHECKING:
    from pymeasure.experiment import Procedure

    from andor_qt.core.hardware_manager import HardwareManager

log = logging.getLogger(__name__)

# Procedure attributes forwarded to camera.apply_camera_settings()
_PROCEDURE_CAMERA_KEYS = (
    "vs_speed_index",
    "hs_speed_index",
    "amplifier_type",
    "em_gain",
    "preamp_gain_index",
)


class ExperimentQueueRunner(QObject):
    """Runs queued procedures sequentially in a background thread.

    Each procedure's parameters (exposure_time, center_wavelength, etc.)
    are used to configure the hardware and acquire data.

    Signals:
        procedure_started: (queue_index, procedure)
        procedure_completed: (queue_index,)
        procedure_failed: (queue_index, error_message)
        queue_progress: (completed_count, total_count)
        queue_completed: ()
        spectrum_ready: (wavelengths, intensities, params_dict)
        image_ready: (image, wavelengths, params_dict)
    """

    procedure_started = Signal(int, object)
    procedure_completed = Signal(int)
    procedure_failed = Signal(int, str)
    queue_progress = Signal(int, int)
    queue_completed = Signal()
    spectrum_ready = Signal(object, object, dict)  # (wavelengths, intensities, params)
    image_ready = Signal(object, object, dict)  # (image, wavelengths, params)

    def __init__(self, hw_manager: HardwareManager, parent: QObject | None = None):
        super().__init__(parent)
        self._hw_manager = hw_manager
        self._queue: Deque[Tuple[int, "Procedure"]] = deque()
        self._next_index: int = 0
        self._is_running: bool = False
        self._abort_requested: bool = False
        self._abort_all_requested: bool = False
        self._thread: Optional[threading.Thread] = None

    def add(self, procedure: "Procedure") -> int:
        """Add a procedure to the queue.

        Args:
            procedure: Configured Procedure instance.

        Returns:
            Queue index for tracking.
        """
        idx = self._next_index
        self._next_index += 1
        self._queue.append((idx, procedure))
        return idx

    def run(self) -> None:
        """Start executing the queue in a background thread."""
        if self._is_running:
            log.warning("Queue is already running")
            return

        if not self._queue:
            self.queue_completed.emit()
            return

        self._is_running = True
        self._abort_requested = False
        self._abort_all_requested = False

        self._thread = threading.Thread(target=self._run_queue, daemon=True)
        self._thread.start()

    def abort(self) -> None:
        """Stop execution after the current procedure finishes."""
        self._abort_requested = True

    def abort_all(self) -> None:
        """Cancel all remaining procedures and stop."""
        self._abort_all_requested = True
        self._abort_requested = True
        self._queue.clear()

    def clear(self) -> None:
        """Clear all pending procedures (does not stop current)."""
        self._queue.clear()

    @property
    def is_running(self) -> bool:
        """Whether the queue is currently executing."""
        return self._is_running

    @property
    def pending_count(self) -> int:
        """Number of procedures still in the queue."""
        return len(self._queue)

    def _run_queue(self) -> None:
        """Execute queued procedures sequentially (runs in bg thread)."""
        total = len(self._queue)
        completed_count = 0

        try:
            while self._queue and not self._abort_requested:
                idx, procedure = self._queue.popleft()

                self.procedure_started.emit(idx, procedure)

                try:
                    self._execute_procedure(procedure)
                    self.procedure_completed.emit(idx)
                except Exception as e:
                    log.error(f"Procedure {idx} failed: {e}")
                    self.procedure_failed.emit(idx, str(e))

                completed_count += 1
                self.queue_progress.emit(completed_count, total)

        finally:
            self._is_running = False
            self.queue_completed.emit()

    def _execute_procedure(self, procedure: "Procedure") -> None:
        """Execute a single procedure using hardware manager.

        Reads the procedure's parameters and performs the acquisition.
        """
        from andor_qt.procedures import ImageProcedure, SpectrumProcedure

        # Set delay position if specified
        if hasattr(procedure, "delay_position"):
            delay_ps = procedure.delay_position
            if self._hw_manager.motion_manager:
                # Get the "delay" axis (or first available)
                axis = self._hw_manager.motion_manager.get_axis("delay")
                if axis:
                    log.info(f"Setting delay position to {delay_ps} ps")
                    axis.position_ps = delay_ps

        # Configure spectrograph
        if hasattr(procedure, "grating"):
            self._hw_manager.spectrograph.grating = procedure.grating
        if hasattr(procedure, "center_wavelength"):
            self._hw_manager.spectrograph.wavelength = procedure.center_wavelength

        # Get hbin for calibration (FVB mode uses hbin, image mode also uses it)
        hbin = getattr(procedure, "hbin", 1)

        # Get calibration with binning factor
        calibration = self._hw_manager.get_calibration(hbin=hbin)

        # Apply camera settings carried by the procedure (amplifier, speeds, gains)
        cam_settings = {
            key: getattr(procedure, key)
            for key in _PROCEDURE_CAMERA_KEYS
            if getattr(procedure, key, None) is not None
        }
        if cam_settings:
            self._hw_manager.camera.apply_camera_settings(cam_settings)

        # Set exposure
        self._hw_manager.camera.set_exposure(procedure.exposure_time)

        if isinstance(procedure, SpectrumProcedure):
            self._acquire_spectrum(procedure, calibration)
        elif isinstance(procedure, ImageProcedure):
            self._acquire_image(procedure, calibration)
        else:
            raise TypeError(f"Unknown procedure type: {type(procedure)}")

    def _acquire_spectrum(self, procedure, calibration: np.ndarray) -> None:
        """Acquire FVB spectrum data."""
        num_accum = getattr(procedure, "num_accumulations", 1)
        hbin = getattr(procedure, "hbin", 1)

        if num_accum > 1:
            eff_xpixels = self._hw_manager.camera.xpixels // hbin
            accumulated = np.zeros(eff_xpixels, dtype=np.float64)
            for i in range(num_accum):
                data = self._hw_manager.camera.acquire_fvb(hbin=hbin)
                accumulated += data
            data = accumulated / num_accum
        else:
            data = self._hw_manager.camera.acquire_fvb(hbin=hbin)

        # Extract procedure parameters for labeling
        params = {
            "exposure_time": getattr(procedure, "exposure_time", 0),
            "center_wavelength": getattr(procedure, "center_wavelength", 0),
            "grating": getattr(procedure, "grating", 1),
            "delay_position": getattr(procedure, "delay_position", 0),
            "num_accumulations": num_accum,
            "hbin": hbin,
        }
        self.spectrum_ready.emit(calibration, data, params)

    def _acquire_image(self, procedure, calibration: np.ndarray) -> None:
        """Acquire 2D image data."""
        hbin = getattr(procedure, "hbin", 1)
        vbin = getattr(procedure, "vbin", 1)

        data = self._hw_manager.camera.acquire_image(hbin=hbin, vbin=vbin)

        # Extract procedure parameters for labeling
        params = {
            "exposure_time": getattr(procedure, "exposure_time", 0),
            "center_wavelength": getattr(procedure, "center_wavelength", 0),
            "grating": getattr(procedure, "grating", 1),
            "delay_position": getattr(procedure, "delay_position", 0),
            "hbin": hbin,
            "vbin": vbin,
        }
        self.image_ready.emit(data, calibration, params)
