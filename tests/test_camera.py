"""Pytest tests for AndorCamera PyMeasure instrument wrapper.

These tests use mock SDK implementations to test the camera interface
without requiring real hardware.

Fixtures (mock_sdk, camera, initialized_camera) are provided by conftest.py.
"""

from __future__ import annotations

import numpy as np
import pytest


class TestAndorCameraInitialization:
    """Tests for camera initialization and shutdown."""

    def test_initialize_success(self, camera):
        """Camera initializes and reads detector info."""
        camera.initialize()

        assert camera._initialized
        assert camera.xpixels == 1024
        assert camera.ypixels == 256
        assert camera.info is not None
        assert camera.info.serial_number == "12345"

    def test_initialize_already_initialized(self, initialized_camera, caplog):
        """Re-initializing logs a warning."""
        initialized_camera.initialize()
        assert "already initialized" in caplog.text.lower()

    def test_shutdown_success(self, initialized_camera):
        """Camera shuts down cleanly."""
        # Warm up first (mock is already warm)
        initialized_camera.shutdown()
        assert not initialized_camera._initialized

    def test_shutdown_not_initialized(self, camera):
        """Shutdown on non-initialized camera is a no-op."""
        camera.shutdown()  # Should not raise
        assert not camera._initialized


class TestAndorCameraTemperature:
    """Tests for temperature control."""

    def test_temperature_property(self, initialized_camera):
        """Temperature returns current value."""
        temp = initialized_camera.temperature
        assert isinstance(temp, float)
        assert temp == 20.0  # Default mock temperature

    def test_temperature_status_off(self, initialized_camera):
        """Temperature status is OFF when cooler is off."""
        status = initialized_camera.temperature_status
        assert status == "OFF"

    def test_cooler_on_sets_target(self, initialized_camera):
        """Cooler ON sets target temperature."""
        initialized_camera.cooler_on(target=-60)
        assert initialized_camera._cooler_on

    def test_cooler_off(self, initialized_camera):
        """Cooler can be turned off."""
        initialized_camera.cooler_on(target=-60)
        initialized_camera.cooler_off()
        assert not initialized_camera._cooler_on

    def test_cooler_on_not_initialized(self, camera):
        """Cooler ON raises if not initialized."""
        with pytest.raises(RuntimeError, match="not initialized"):
            camera.cooler_on(target=-60)


class TestAndorCameraExposure:
    """Tests for exposure time setting."""

    def test_set_exposure(self, initialized_camera):
        """Exposure time can be set."""
        initialized_camera.set_exposure(0.5)
        # The mock stores the value - we'd need to read it back through SDK

    def test_set_exposure_not_initialized(self, camera):
        """Set exposure raises if not initialized."""
        with pytest.raises(RuntimeError, match="not initialized"):
            camera.set_exposure(0.1)


class TestAndorCameraAcquisitionFVB:
    """Tests for FVB (Full Vertical Binning) acquisition."""

    def test_acquire_fvb_returns_1d_array(self, initialized_camera):
        """FVB acquisition returns 1D numpy array."""
        initialized_camera.set_exposure(0.01)  # Short exposure for test
        data = initialized_camera.acquire_fvb()

        assert isinstance(data, np.ndarray)
        assert data.ndim == 1
        assert len(data) == initialized_camera.xpixels

    def test_acquire_fvb_data_reasonable(self, initialized_camera):
        """FVB data has reasonable values."""
        initialized_camera.set_exposure(0.01)
        data = initialized_camera.acquire_fvb()

        assert np.all(data >= 0)  # No negative values
        assert np.max(data) > 0  # Has signal

    def test_acquire_fvb_not_initialized(self, camera):
        """FVB acquisition raises if not initialized."""
        with pytest.raises(RuntimeError, match="not initialized"):
            camera.acquire_fvb()

    def test_acquire_fvb_with_hbin(self, initialized_camera):
        """FVB acquisition with horizontal binning returns reduced pixel count."""
        initialized_camera.set_exposure(0.01)
        hbin = 2
        data = initialized_camera.acquire_fvb(hbin=hbin)

        assert isinstance(data, np.ndarray)
        assert data.ndim == 1
        assert len(data) == initialized_camera.xpixels // hbin

    def test_acquire_fvb_with_hbin_4(self, initialized_camera):
        """FVB with hbin=4 returns 1/4 pixel count."""
        initialized_camera.set_exposure(0.01)
        hbin = 4
        data = initialized_camera.acquire_fvb(hbin=hbin)

        assert len(data) == initialized_camera.xpixels // hbin

    def test_acquire_fvb_invalid_hbin(self, initialized_camera):
        """FVB with invalid hbin raises ValueError."""
        initialized_camera.set_exposure(0.01)
        # 3 is not a factor of 1024
        with pytest.raises(ValueError, match="must be a factor"):
            initialized_camera.acquire_fvb(hbin=3)


class TestAndorCameraAcquisitionImage:
    """Tests for 2D image acquisition."""

    def test_acquire_image_returns_2d_array(self, initialized_camera):
        """Image acquisition returns 2D numpy array."""
        initialized_camera.set_exposure(0.01)
        data = initialized_camera.acquire_image(hbin=1, vbin=1)

        assert isinstance(data, np.ndarray)
        assert data.ndim == 2
        assert data.shape == (initialized_camera.ypixels, initialized_camera.xpixels)

    def test_acquire_image_with_binning(self, initialized_camera):
        """Image acquisition with binning returns correct shape."""
        initialized_camera.set_exposure(0.01)
        data = initialized_camera.acquire_image(hbin=2, vbin=4)

        expected_x = initialized_camera.xpixels // 2
        expected_y = initialized_camera.ypixels // 4
        assert data.shape == (expected_y, expected_x)

    def test_acquire_image_invalid_hbin(self, initialized_camera):
        """Invalid hbin raises ValueError."""
        with pytest.raises(ValueError, match="hbin"):
            initialized_camera.acquire_image(hbin=3, vbin=1)  # 1024 % 3 != 0

    def test_acquire_image_invalid_vbin(self, initialized_camera):
        """Invalid vbin raises ValueError."""
        with pytest.raises(ValueError, match="vbin"):
            initialized_camera.acquire_image(hbin=1, vbin=5)  # 256 % 5 != 0

    def test_acquire_image_not_initialized(self, camera):
        """Image acquisition raises if not initialized."""
        with pytest.raises(RuntimeError, match="not initialized"):
            camera.acquire_image()


class TestAndorCameraAbort:
    """Tests for acquisition abort."""

    def test_abort_acquisition(self, initialized_camera):
        """Abort acquisition does not raise."""
        initialized_camera.abort_acquisition()

    def test_abort_not_initialized(self, camera):
        """Abort on non-initialized camera is a no-op."""
        camera.abort_acquisition()  # Should not raise


class TestAndorCameraWarmup:
    """Tests for warmup before shutdown."""

    def test_warmup_already_warm(self, initialized_camera):
        """Warmup returns True if already warm."""
        # Mock starts at 20C, target is -20C
        result = initialized_camera.warmup(target=-20, timeout=1.0)
        assert result

    def test_warmup_not_initialized(self, camera):
        """Warmup on non-initialized camera returns True."""
        result = camera.warmup()
        assert result


class TestAndorCameraInfo:
    """Tests for camera info properties."""

    def test_xpixels_not_initialized(self, camera):
        """xpixels returns 0 if not initialized."""
        assert camera.xpixels == 0

    def test_ypixels_not_initialized(self, camera):
        """ypixels returns 0 if not initialized."""
        assert camera.ypixels == 0

    def test_info_not_initialized(self, camera):
        """info returns None if not initialized."""
        assert camera.info is None

    def test_info_after_init(self, initialized_camera):
        """info is populated after initialization."""
        info = initialized_camera.info
        assert info is not None
        assert info.xpixels == 1024
        assert info.ypixels == 256
        assert info.pixel_width == 26.0
        assert info.pixel_height == 26.0


class TestAndorCameraTemperatureNotInitialized:
    """Tests for temperature when not initialized."""

    def test_temperature_not_initialized(self, camera):
        """Temperature returns 20.0 if not initialized."""
        assert camera.temperature == 20.0

    def test_temperature_status_not_initialized(self, camera):
        """Temperature status returns NOT_INITIALIZED if not initialized."""
        assert camera.temperature_status == "NOT_INITIALIZED"


class TestMockCameraSDKSettings:
    """Tests for mock SDK camera settings methods (VS/HS speed, amplifier, gain)."""

    def test_mock_get_vs_speeds_returns_5_options(self, initialized_camera):
        """Mock VS speed query returns 5 options."""
        speeds = initialized_camera.get_vs_speeds()
        assert len(speeds) == 5

    def test_mock_vs_speeds_have_correct_values(self, initialized_camera):
        """Mock VS speeds match DU970P hardware values."""
        speeds = initialized_camera.get_vs_speeds()
        us_values = [s[1] for s in speeds]
        assert us_values == pytest.approx([4.9, 9.8, 19.0, 38.0, 57.0], rel=0.01)

    def test_mock_set_vs_speed_valid_index(self, initialized_camera):
        """set_vs_speed accepts valid indices 0-4."""
        for i in range(5):
            initialized_camera.set_vs_speed(i)  # Should not raise

    def test_mock_set_vs_speed_not_initialized_raises(self, camera):
        """set_vs_speed raises if not initialized."""
        with pytest.raises(RuntimeError, match="not initialized"):
            camera.set_vs_speed(0)

    def test_mock_get_hs_speeds_em_amplifier(self, initialized_camera):
        """HS speeds for EM amplifier (type=0) returns 3 options."""
        speeds = initialized_camera.get_hs_speeds(amplifier_type=0)
        assert len(speeds) == 3

    def test_mock_get_hs_speeds_conventional_amplifier(self, initialized_camera):
        """HS speeds for conventional amplifier (type=1) returns 3 options."""
        speeds = initialized_camera.get_hs_speeds(amplifier_type=1)
        assert len(speeds) == 3

    def test_mock_hs_speeds_values(self, initialized_camera):
        """HS speed values are 3.0, 1.0, 0.05 MHz."""
        speeds = initialized_camera.get_hs_speeds(amplifier_type=0)
        mhz_values = [s[1] for s in speeds]
        assert mhz_values == pytest.approx([3.0, 1.0, 0.05], rel=0.01)

    def test_mock_set_hs_speed_not_initialized_raises(self, camera):
        """set_hs_speed raises if not initialized."""
        with pytest.raises(RuntimeError, match="not initialized"):
            camera.set_hs_speed(amplifier_type=0, index=0)

    def test_mock_set_amplifier_em(self, initialized_camera):
        """set_amplifier(0) selects EM output."""
        initialized_camera.set_amplifier(0)  # Should not raise

    def test_mock_set_amplifier_conventional(self, initialized_camera):
        """set_amplifier(1) selects conventional output."""
        initialized_camera.set_amplifier(1)  # Should not raise

    def test_mock_set_amplifier_not_initialized_raises(self, camera):
        """set_amplifier raises if not initialized."""
        with pytest.raises(RuntimeError, match="not initialized"):
            camera.set_amplifier(0)

    def test_mock_set_em_gain_valid(self, initialized_camera):
        """set_em_gain accepts integer in valid range."""
        initialized_camera.set_em_gain(100)  # Should not raise

    def test_mock_set_em_gain_not_initialized_raises(self, camera):
        """set_em_gain raises if not initialized."""
        with pytest.raises(RuntimeError, match="not initialized"):
            camera.set_em_gain(100)

    def test_mock_get_preamp_gains_returns_list(self, initialized_camera):
        """get_preamp_gains returns a list of (index, gain) tuples."""
        gains = initialized_camera.get_preamp_gains()
        assert len(gains) >= 1
        assert all(len(g) == 2 for g in gains)

    def test_mock_set_preamp_gain_valid(self, initialized_camera):
        """set_preamp_gain accepts valid index."""
        initialized_camera.set_preamp_gain(0)  # Should not raise

    def test_mock_set_preamp_gain_not_initialized_raises(self, camera):
        """set_preamp_gain raises if not initialized."""
        with pytest.raises(RuntimeError, match="not initialized"):
            camera.set_preamp_gain(0)

    def test_mock_set_single_track(self, initialized_camera):
        """set_single_track accepts centre and height."""
        initialized_camera.set_single_track(centre=100, height=10)  # Should not raise

    def test_mock_set_single_track_not_initialized_raises(self, camera):
        """set_single_track raises if not initialized."""
        with pytest.raises(RuntimeError, match="not initialized"):
            camera.set_single_track(centre=100, height=10)

    def test_mock_set_crop_mode_active(self, initialized_camera):
        """set_crop_mode enables crop mode."""
        initialized_camera.set_crop_mode(active=True, crop_height=20, crop_width=1600, vbin=1, hbin=1)

    def test_mock_set_crop_mode_inactive(self, initialized_camera):
        """set_crop_mode disables crop mode."""
        initialized_camera.set_crop_mode(active=False, crop_height=20, crop_width=1600, vbin=1, hbin=1)

    def test_mock_set_crop_mode_not_initialized_raises(self, camera):
        """set_crop_mode raises if not initialized."""
        with pytest.raises(RuntimeError, match="not initialized"):
            camera.set_crop_mode(active=True, crop_height=20, crop_width=1600, vbin=1, hbin=1)

    def test_apply_camera_settings_applies_all(self, initialized_camera):
        """apply_camera_settings applies a full settings dict without error."""
        settings = {
            "vs_speed_index": 1,
            "hs_speed_index": 0,
            "amplifier_type": 0,
            "em_gain": 50,
            "preamp_gain_index": 1,
        }
        initialized_camera.apply_camera_settings(settings)  # Should not raise

    def test_apply_camera_settings_not_initialized_raises(self, camera):
        """apply_camera_settings raises if not initialized."""
        with pytest.raises(RuntimeError, match="not initialized"):
            camera.apply_camera_settings({"vs_speed_index": 0})


class TestEMGainMode:
    """SetEMGainMode(3) at init makes the integer passed to SetEMCCDGain the
    actual electron multiplication factor (instead of an opaque DAC code).
    Reproducibility across sessions and HDF5 metadata becomes meaningful."""

    def test_initialize_sets_em_gain_mode_to_real_em_gain(self, camera):
        """After initialize(), the camera is in Real EM Gain mode (mode 3)."""
        camera.initialize()
        assert camera.em_gain_mode == 3

    def test_initialize_records_real_em_gain_range(self, camera):
        """GetEMGainRange is queried AFTER SetEMGainMode(3), so the recorded
        range is the physical-gain envelope (mode-3) rather than 8-bit DAC."""
        camera.initialize()
        low, high = camera.info.em_gain_range
        # Mode 3 returns the physical envelope (mock returns (1, 300));
        # mode 0 would return (0, 255). Anything other than (1, 300) means
        # GetEMGainRange ran before SetEMGainMode.
        assert (low, high) == (1, 300)

    def test_initialize_falls_back_when_set_mode_fails(self, camera, caplog):
        """If SetEMGainMode fails, em_gain_mode falls back to 0 with a warning."""
        # Patch the SDK's SetEMGainMode to return a non-success code
        original_initialize = camera.initialize

        def _patched_initialize():
            # Install patch BEFORE initialize triggers any SDK imports
            from andor_pymeasure.instruments.mock import MockAtmcd
            MockAtmcd.SetEMGainMode = lambda self, mode: 20992  # DRV_NOT_AVAILABLE
            try:
                original_initialize()
            finally:
                # restore
                def real(self, mode):
                    if mode not in (0, 1, 2, 3):
                        return 20066
                    self._state.em_gain_mode = mode
                    return 20002
                MockAtmcd.SetEMGainMode = real

        _patched_initialize()
        assert camera.em_gain_mode == 0
        assert "SetEMGainMode" in caplog.text

    def test_get_em_gain_returns_set_value(self, initialized_camera):
        """get_em_gain() reads back the integer that SetEMCCDGain stored."""
        initialized_camera.set_em_gain(50)
        assert initialized_camera.get_em_gain() == 50

    def test_get_em_gain_not_initialized_returns_zero(self, camera):
        """get_em_gain() on uninitialized camera returns 0, does not raise."""
        assert camera.get_em_gain() == 0

    def test_apply_camera_settings_warns_on_em_gain_mismatch(
        self, initialized_camera, caplog
    ):
        """If the readback after SetEMCCDGain differs from the request, warn."""
        # Force the mock to silently store a different gain on the next set.
        from andor_pymeasure.instruments.mock import MockAtmcd

        original_set = MockAtmcd.SetEMCCDGain

        def _mismatching_set(self, gain):
            self._state.em_gain = gain + 7  # off by 7
            return 20002

        MockAtmcd.SetEMCCDGain = _mismatching_set
        try:
            initialized_camera.apply_camera_settings({
                "amplifier_type": 0,
                "em_gain": 50,
            })
        finally:
            MockAtmcd.SetEMCCDGain = original_set

        msg = caplog.text.lower()
        assert "em gain" in msg
        assert ("readback" in msg) or ("mismatch" in msg)
