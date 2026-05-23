"""Tests for HardwareManager.

These tests verify the HardwareManager singleton correctly manages
camera and spectrograph instances using mock hardware.
"""

from __future__ import annotations

import threading
import time
from typing import Any
from unittest.mock import MagicMock, patch

import pytest


class TestHardwareManagerSingleton:
    """Tests for HardwareManager singleton pattern."""

    def test_instance_returns_same_object(self, reset_hardware_manager):
        """HardwareManager.instance() returns the same object."""
        HardwareManager = reset_hardware_manager

        manager1 = HardwareManager.instance()
        manager2 = HardwareManager.instance()

        assert manager1 is manager2

    def test_constructor_returns_same_object(self, reset_hardware_manager):
        """HardwareManager() returns the same singleton."""
        HardwareManager = reset_hardware_manager

        manager1 = HardwareManager()
        manager2 = HardwareManager()

        assert manager1 is manager2

    def test_reset_instance_clears_singleton(self, reset_hardware_manager):
        """reset_instance() allows creating a new singleton."""
        HardwareManager = reset_hardware_manager

        manager1 = HardwareManager.instance()
        id1 = id(manager1)

        HardwareManager.reset_instance()

        manager2 = HardwareManager.instance()
        id2 = id(manager2)

        assert id1 != id2

    def test_thread_safe_singleton(self, reset_hardware_manager):
        """Singleton creation is thread-safe."""
        HardwareManager = reset_hardware_manager

        instances = []
        errors = []

        def get_instance():
            try:
                manager = HardwareManager.instance()
                instances.append(manager)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=get_instance) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(instances) == 10
        # All should be the same instance
        assert all(inst is instances[0] for inst in instances)


class TestHardwareManagerProperties:
    """Tests for HardwareManager property accessors."""

    def test_camera_none_before_init(self, hardware_manager):
        """camera property is None before initialization."""
        assert hardware_manager.camera is None

    def test_spectrograph_none_before_init(self, hardware_manager):
        """spectrograph property is None before initialization."""
        assert hardware_manager.spectrograph is None

    def test_is_initialized_false_before_init(self, hardware_manager):
        """is_initialized is False before initialization."""
        assert hardware_manager.is_initialized is False

    def test_mock_mode_enabled(self, hardware_manager):
        """mock_mode is True when ANDOR_MOCK env var is set."""
        assert hardware_manager.mock_mode is True


class TestHardwareManagerInitialize:
    """Tests for HardwareManager.initialize()."""

    def test_initialize_creates_camera(self, hardware_manager, wait_for):
        """Initialize creates camera instance."""
        completed = []
        hardware_manager.initialize(on_complete=lambda: completed.append(True))

        wait_for(lambda: len(completed) > 0)

        assert hardware_manager.camera is not None

    def test_initialize_creates_spectrograph(self, hardware_manager, wait_for):
        """Initialize creates spectrograph instance."""
        completed = []
        hardware_manager.initialize(on_complete=lambda: completed.append(True))

        wait_for(lambda: len(completed) > 0)

        assert hardware_manager.spectrograph is not None

    def test_initialize_sets_is_initialized(self, hardware_manager, wait_for):
        """Initialize sets is_initialized to True."""
        completed = []
        hardware_manager.initialize(on_complete=lambda: completed.append(True))

        wait_for(lambda: len(completed) > 0)

        assert hardware_manager.is_initialized is True

    def test_initialize_calls_on_complete(self, hardware_manager, wait_for):
        """Initialize calls on_complete callback."""
        completed = []
        hardware_manager.initialize(on_complete=lambda: completed.append(True))

        assert wait_for(lambda: len(completed) > 0)
        assert completed == [True]

    def test_initialize_skips_if_already_initialized(self, hardware_manager, wait_for):
        """Initialize skips if already initialized."""
        completed = []
        hardware_manager.initialize(on_complete=lambda: completed.append(1))
        wait_for(lambda: len(completed) > 0)

        # Second call
        hardware_manager.initialize(on_complete=lambda: completed.append(2))
        wait_for(lambda: len(completed) > 1)

        # Both callbacks called (second is immediate)
        assert 1 in completed
        assert 2 in completed


class TestHardwareManagerSetCooler:
    """Tests for HardwareManager.set_cooler()."""

    def test_set_cooler_on(self, hardware_manager, wait_for):
        """set_cooler turns cooler on."""
        completed = []
        hardware_manager.initialize(on_complete=lambda: completed.append(True))
        wait_for(lambda: len(completed) > 0)

        cooler_done = []
        hardware_manager.set_cooler(on=True, target_temp=-70, on_complete=lambda: cooler_done.append(True))
        wait_for(lambda: len(cooler_done) > 0)

        assert hardware_manager.camera._cooler_on is True

    def test_set_cooler_off(self, hardware_manager, wait_for):
        """set_cooler turns cooler off."""
        completed = []
        hardware_manager.initialize(on_complete=lambda: completed.append(True))
        wait_for(lambda: len(completed) > 0)

        # Turn on first
        cooler_done = []
        hardware_manager.set_cooler(on=True, on_complete=lambda: cooler_done.append(True))
        wait_for(lambda: len(cooler_done) > 0)

        # Turn off
        cooler_done.clear()
        hardware_manager.set_cooler(on=False, on_complete=lambda: cooler_done.append(True))
        wait_for(lambda: len(cooler_done) > 0)

        assert hardware_manager.camera._cooler_on is False

    def test_set_cooler_without_camera(self, hardware_manager):
        """set_cooler does nothing if camera not initialized."""
        # Should not raise
        hardware_manager.set_cooler(on=True)


class TestHardwareManagerSetGrating:
    """Tests for HardwareManager.set_grating()."""

    def test_set_grating(self, hardware_manager, wait_for):
        """set_grating changes spectrograph grating."""
        completed = []
        hardware_manager.initialize(on_complete=lambda: completed.append(True))
        wait_for(lambda: len(completed) > 0)

        grating_done = []
        hardware_manager.set_grating(2, on_complete=lambda: grating_done.append(True))
        wait_for(lambda: len(grating_done) > 0)

        assert hardware_manager.spectrograph.grating == 2

    def test_set_grating_without_spectrograph(self, hardware_manager):
        """set_grating does nothing if spectrograph not initialized."""
        # Should not raise
        hardware_manager.set_grating(1)


class TestHardwareManagerSetWavelength:
    """Tests for HardwareManager.set_wavelength()."""

    def test_set_wavelength(self, hardware_manager, wait_for):
        """set_wavelength changes spectrograph wavelength."""
        completed = []
        hardware_manager.initialize(on_complete=lambda: completed.append(True))
        wait_for(lambda: len(completed) > 0)

        wavelength_done = []
        hardware_manager.set_wavelength(600.0, on_complete=lambda: wavelength_done.append(True))
        wait_for(lambda: len(wavelength_done) > 0)

        assert hardware_manager.spectrograph.wavelength == 600.0

    def test_set_wavelength_without_spectrograph(self, hardware_manager):
        """set_wavelength does nothing if spectrograph not initialized."""
        # Should not raise
        hardware_manager.set_wavelength(500.0)


class TestHardwareManagerCalibration:
    """Tests for HardwareManager.get_calibration()."""

    def test_get_calibration(self, hardware_manager, wait_for):
        """get_calibration returns wavelength array."""
        completed = []
        hardware_manager.initialize(on_complete=lambda: completed.append(True))
        wait_for(lambda: len(completed) > 0)

        calibration = hardware_manager.get_calibration()

        assert calibration is not None
        assert len(calibration) == hardware_manager.camera.xpixels

    def test_get_calibration_without_hardware(self, hardware_manager):
        """get_calibration returns None if hardware not initialized."""
        result = hardware_manager.get_calibration()
        assert result is None


class TestHardwareManagerShutdown:
    """Tests for HardwareManager.shutdown()."""

    def test_shutdown_clears_camera(self, hardware_manager, wait_for):
        """shutdown clears camera reference."""
        completed = []
        hardware_manager.initialize(on_complete=lambda: completed.append(True))
        wait_for(lambda: len(completed) > 0)

        shutdown_done = []
        hardware_manager.shutdown(
            warmup=False,  # Skip warmup for faster tests
            on_complete=lambda: shutdown_done.append(True)
        )
        wait_for(lambda: len(shutdown_done) > 0)

        assert hardware_manager.camera is None

    def test_shutdown_clears_spectrograph(self, hardware_manager, wait_for):
        """shutdown clears spectrograph reference."""
        completed = []
        hardware_manager.initialize(on_complete=lambda: completed.append(True))
        wait_for(lambda: len(completed) > 0)

        shutdown_done = []
        hardware_manager.shutdown(
            warmup=False,
            on_complete=lambda: shutdown_done.append(True)
        )
        wait_for(lambda: len(shutdown_done) > 0)

        assert hardware_manager.spectrograph is None

    def test_shutdown_sets_is_initialized_false(self, hardware_manager, wait_for):
        """shutdown sets is_initialized to False."""
        completed = []
        hardware_manager.initialize(on_complete=lambda: completed.append(True))
        wait_for(lambda: len(completed) > 0)

        shutdown_done = []
        hardware_manager.shutdown(
            warmup=False,
            on_complete=lambda: shutdown_done.append(True)
        )
        wait_for(lambda: len(shutdown_done) > 0)

        assert hardware_manager.is_initialized is False

    def test_shutdown_without_init(self, hardware_manager, wait_for):
        """shutdown does nothing if not initialized."""
        completed = []
        hardware_manager.shutdown(on_complete=lambda: completed.append(True))

        # Should complete immediately
        wait_for(lambda: len(completed) > 0, timeout=1.0)
        assert completed == [True]

    def test_shutdown_reports_progress(self, hardware_manager, wait_for):
        """shutdown calls on_progress with status messages."""
        completed = []
        hardware_manager.initialize(on_complete=lambda: completed.append(True))
        wait_for(lambda: len(completed) > 0)

        progress_messages = []
        shutdown_done = []
        hardware_manager.shutdown(
            warmup=False,
            on_complete=lambda: shutdown_done.append(True),
            on_progress=lambda msg: progress_messages.append(msg)
        )
        wait_for(lambda: len(shutdown_done) > 0)

        assert len(progress_messages) > 0
        # Should contain camera/spectrograph shutdown messages
        assert any("camera" in msg.lower() for msg in progress_messages)


class TestHardwareManagerInjectIntoProcedure:
    """Tests for HardwareManager.inject_into_procedure()."""

    def test_inject_into_procedure(self, hardware_manager, wait_for):
        """inject_into_procedure sets shared hardware on procedure class."""
        completed = []
        hardware_manager.initialize(on_complete=lambda: completed.append(True))
        wait_for(lambda: len(completed) > 0)

        class MockProcedure:
            pass

        hardware_manager.inject_into_procedure(MockProcedure)

        assert MockProcedure._shared_camera is hardware_manager.camera
        assert MockProcedure._shared_spectrograph is hardware_manager.spectrograph


class TestHardwareManagerEventBus:
    """Tests for HardwareManager EventBus integration."""

    def test_initialize_publishes_event(
        self, hardware_manager, wait_for, reset_event_bus, handler_factory
    ):
        """Initialize publishes hardware.initialized event."""
        from andor_qt.core.event_bus import get_event_bus

        bus = get_event_bus()
        handler = handler_factory("init_handler")
        bus.subscribe("hardware.initialized", handler)

        completed = []
        hardware_manager.initialize(on_complete=lambda: completed.append(True))
        wait_for(lambda: len(completed) > 0)

        handler.assert_called_once()
        call_args = handler.call_args
        assert "camera" in call_args.kwargs
        assert "spectrograph" in call_args.kwargs

    def test_set_grating_publishes_event(
        self, hardware_manager, wait_for, reset_event_bus, handler_factory
    ):
        """set_grating publishes hardware.grating_changed event."""
        from andor_qt.core.event_bus import get_event_bus

        # Initialize first
        completed = []
        hardware_manager.initialize(on_complete=lambda: completed.append(True))
        wait_for(lambda: len(completed) > 0)

        bus = get_event_bus()
        handler = handler_factory("grating_handler")
        bus.subscribe("hardware.grating_changed", handler)

        grating_done = []
        hardware_manager.set_grating(2, on_complete=lambda: grating_done.append(True))
        wait_for(lambda: len(grating_done) > 0)

        handler.assert_called_once()
        call_args = handler.call_args
        assert call_args.kwargs["grating"] == 2

    def test_set_wavelength_publishes_event(
        self, hardware_manager, wait_for, reset_event_bus, handler_factory
    ):
        """set_wavelength publishes hardware.wavelength_changed event."""
        from andor_qt.core.event_bus import get_event_bus

        # Initialize first
        completed = []
        hardware_manager.initialize(on_complete=lambda: completed.append(True))
        wait_for(lambda: len(completed) > 0)

        bus = get_event_bus()
        handler = handler_factory("wavelength_handler")
        bus.subscribe("hardware.wavelength_changed", handler)

        wavelength_done = []
        hardware_manager.set_wavelength(
            600.0, on_complete=lambda: wavelength_done.append(True)
        )
        wait_for(lambda: len(wavelength_done) > 0)

        handler.assert_called_once()
        call_args = handler.call_args
        assert call_args.kwargs["wavelength"] == 600.0

    def test_shutdown_publishes_event(
        self, hardware_manager, wait_for, reset_event_bus, handler_factory
    ):
        """shutdown publishes hardware.shutdown event."""
        from andor_qt.core.event_bus import get_event_bus

        # Initialize first
        completed = []
        hardware_manager.initialize(on_complete=lambda: completed.append(True))
        wait_for(lambda: len(completed) > 0)

        bus = get_event_bus()
        handler = handler_factory("shutdown_handler")
        bus.subscribe("hardware.shutdown", handler)

        shutdown_done = []
        hardware_manager.shutdown(
            warmup=False, on_complete=lambda: shutdown_done.append(True)
        )
        wait_for(lambda: len(shutdown_done) > 0)

        handler.assert_called_once()


class TestHardwareManagerChopperPair:
    """Tests for HardwareManager chopper-pair integration."""

    def test_chopper_pair_none_before_init(self, hardware_manager):
        """chopper_pair is None before initialization."""
        assert hardware_manager.chopper_pair is None

    def test_initialize_creates_mock_chopper_pair(self, hardware_manager, wait_for):
        """Initialize creates a ChopperPair with mock choppers in mock mode."""
        from andor_pymeasure.instruments.thorlabs_mc2000b import MockMC2000B

        completed = []
        hardware_manager.initialize(on_complete=lambda: completed.append(True))
        wait_for(lambda: len(completed) > 0)

        pair = hardware_manager.chopper_pair
        assert pair is not None
        assert isinstance(pair.pump, MockMC2000B)
        assert isinstance(pair.probe, MockMC2000B)

    def test_chopper_ports_from_config(self, hardware_manager, wait_for):
        """Chopper ports come from AppConfig.choppers."""
        from andor_qt.core.config import (
            AppConfig,
            ChopperConfig,
            HardwareConfig,
        )

        cfg = AppConfig(
            hardware=HardwareConfig(mock_mode=True),
            motion_controllers={
                "enabled": True,
                "controllers": [{"name": "mock", "type": "mock", "axes": []}],
            },
            choppers=ChopperConfig(pump_port="COM_TEST_PUMP", probe_port="COM_TEST_PROBE"),
        )
        hardware_manager.set_config(cfg)

        completed = []
        hardware_manager.initialize(on_complete=lambda: completed.append(True))
        wait_for(lambda: len(completed) > 0)

        pair = hardware_manager.chopper_pair
        assert pair is not None
        assert pair.pump.port == "COM_TEST_PUMP"
        assert pair.probe.port == "COM_TEST_PROBE"

    def test_chopper_disabled_in_config_means_none(self, hardware_manager, wait_for):
        """If ChopperConfig.enabled is False, chopper_pair stays None."""
        from andor_qt.core.config import (
            AppConfig,
            ChopperConfig,
            HardwareConfig,
        )

        cfg = AppConfig(
            hardware=HardwareConfig(mock_mode=True),
            motion_controllers={
                "enabled": True,
                "controllers": [{"name": "mock", "type": "mock", "axes": []}],
            },
            choppers=ChopperConfig(enabled=False),
        )
        hardware_manager.set_config(cfg)

        completed = []
        hardware_manager.initialize(on_complete=lambda: completed.append(True))
        wait_for(lambda: len(completed) > 0)

        assert hardware_manager.chopper_pair is None

    def test_shutdown_disables_choppers(self, hardware_manager, wait_for):
        """Shutdown calls .shutdown() on the chopper pair (choppers disabled)."""
        completed = []
        hardware_manager.initialize(on_complete=lambda: completed.append(True))
        wait_for(lambda: len(completed) > 0)

        pair = hardware_manager.chopper_pair
        assert pair is not None
        # Force enabled so we can confirm shutdown disables them
        pair.pump.enabled = True
        pair.probe.enabled = True

        shutdown_done = []
        hardware_manager.shutdown(
            warmup=False, on_complete=lambda: shutdown_done.append(True)
        )
        wait_for(lambda: len(shutdown_done) > 0)

        # After shutdown the manager clears its reference; we already captured pair
        assert pair.pump.enabled is False
        assert pair.probe.enabled is False
        assert hardware_manager.chopper_pair is None
