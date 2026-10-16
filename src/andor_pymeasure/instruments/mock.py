"""Mock SDK implementations for testing PyMeasure instruments.

This module provides mock implementations of pyAndorSDK2 and pyAndorSpectrograph
that can be used for testing without real hardware.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Tuple
from unittest.mock import MagicMock

import numpy as np


# SDK return codes
DRV_SUCCESS = 20002
DRV_TEMPERATURE_STABILIZED = 20036
DRV_TEMPERATURE_NOT_REACHED = 20037
DRV_TEMPERATURE_NOT_STABILIZED = 20035
DRV_TEMPERATURE_OFF = 20034
DRV_TEMPERATURE_DRIFT = 20038
DRV_ACQUIRING = 20072

ATSPECTROGRAPH_SUCCESS = 0

# Shared mock delay (ps) — updated by MockAxis.position_ps setter so the
# mock camera can produce delay-dependent pump-probe spectra.
_mock_delay_ps: float = 0.0


@dataclass
class MockCameraState:
    """Mock camera internal state."""

    initialized: bool = False
    xpixels: int = 1024
    ypixels: int = 256
    pixel_width: float = 26.0
    pixel_height: float = 26.0
    serial_number: int = 12345
    temperature: float = 20.0
    target_temperature: int = -60
    cooler_on: bool = False
    exposure_time: float = 0.1
    read_mode: int = 0  # 0=FVB, 4=Image
    acquisition_mode: int = 1  # 1=Single scan
    hbin: int = 1
    vbin: int = 1
    acquiring: bool = False
    # Camera settings (DU970P EMCCD)
    vs_speed_index: int = 1         # default: 9.8 µs/pixel (optimum CTE)
    hs_speed_index: int = 0         # default: 3 MHz
    amplifier_type: int = 0         # 0=EM, 1=conventional
    em_gain: int = 1
    em_gain_mode: int = 0   # 0=8-bit DAC, 1=12-bit DAC, 2=linear, 3=Real EM Gain
    preamp_gain_index: int = 0
    single_track_centre: int = 128
    single_track_height: int = 10
    crop_mode_active: bool = False
    crop_height: int = 16
    crop_width: int = 1024
    trigger_mode: int = 0       # 0=internal, 1=external
    fast_ext_trigger: int = 0  # 0=disabled, 1=enabled (SetFastExtTrigger)
    keep_cleans: int = 1       # 0=disabled, 1=enabled (EnableKeepCleans)


@dataclass
class MockSpectrographState:
    """Mock spectrograph internal state."""

    initialized: bool = False
    num_devices: int = 1
    serial_number: str = "SPC-001"
    num_gratings: int = 2
    current_grating: int = 1
    current_wavelength: float = 500.0
    num_pixels: int = 1024
    pixel_width: float = 26.0
    gratings: List[dict] = field(default_factory=lambda: [
        {"lines": 150.0, "blaze": "500nm", "wl_min": 200.0, "wl_max": 1100.0},
        {"lines": 600.0, "blaze": "500nm", "wl_min": 300.0, "wl_max": 800.0},
    ])


class MockAtmcdErrors:
    """Mock atmcd_errors module."""

    class Error_Codes:
        DRV_SUCCESS = DRV_SUCCESS
        DRV_TEMPERATURE_STABILIZED = DRV_TEMPERATURE_STABILIZED
        DRV_TEMPERATURE_NOT_REACHED = DRV_TEMPERATURE_NOT_REACHED
        DRV_TEMPERATURE_NOT_STABILIZED = DRV_TEMPERATURE_NOT_STABILIZED
        DRV_TEMPERATURE_OFF = DRV_TEMPERATURE_OFF
        DRV_TEMPERATURE_DRIFT = DRV_TEMPERATURE_DRIFT
        DRV_ACQUIRING = DRV_ACQUIRING


class MockAtmcdCodes:
    """Mock atmcd_codes module."""

    class Acquisition_Mode:
        SINGLE_SCAN = 1
        ACCUMULATE = 2
        KINETICS = 3
        FAST_KINETICS = 4
        RUN_TILL_ABORT = 5

    class Read_Mode:
        FULL_VERTICAL_BINNING = 0
        MULTI_TRACK = 1
        RANDOM_TRACK = 2
        SINGLE_TRACK = 3
        IMAGE = 4

    class Trigger_Mode:
        INTERNAL = 0
        EXTERNAL = 1


class MockAtmcd:
    """Mock atmcd SDK for camera control.

    This class simulates the Andor SDK2 camera interface for testing.
    """

    def __init__(self, sdk_path: str = ""):
        import threading
        self._state = MockCameraState()
        self._sdk_path = sdk_path
        self._last_temp_check = time.time()
        self._shot_counter: int = 0  # alternates pump-on / pump-off
        self._lock = threading.Lock()

    def Initialize(self, path: str) -> int:
        """Initialize the camera SDK."""
        self._state.initialized = True
        return DRV_SUCCESS

    def ShutDown(self) -> int:
        """Shutdown the camera SDK."""
        self._state.initialized = False
        return DRV_SUCCESS

    def GetCameraSerialNumber(self) -> Tuple[int, int]:
        """Get camera serial number."""
        return (DRV_SUCCESS, self._state.serial_number)

    def GetDetector(self) -> Tuple[int, int, int]:
        """Get detector dimensions."""
        return (DRV_SUCCESS, self._state.xpixels, self._state.ypixels)

    def GetPixelSize(self) -> Tuple[int, float, float]:
        """Get pixel dimensions in micrometers."""
        return (DRV_SUCCESS, self._state.pixel_width, self._state.pixel_height)

    def GetEMGainRange(self) -> Tuple[int, int, int]:
        """Get EM gain range for the currently selected EM gain mode.

        Per the Andor SDK, the returned range depends on the mode set via
        ``SetEMGainMode`` and the sensor temperature. The mock returns
        canonical values for each documented mode.
        """
        mode = self._state.em_gain_mode
        if mode == 0:
            return (DRV_SUCCESS, 0, 255)        # 8-bit DAC
        if mode == 1:
            return (DRV_SUCCESS, 0, 4095)       # 12-bit DAC
        # Modes 2 (linear) and 3 (Real EM Gain): physical gain envelope.
        # Real cameras report ~(1, 250-300) at -65 C; we pick a stable value
        # so tests can assert against it.
        return (DRV_SUCCESS, 1, 300)

    def SetEMGainMode(self, mode: int) -> int:
        """Set EM gain interpretation mode.

        0 = 8-bit DAC (0-255), 1 = 12-bit DAC (0-4095),
        2 = linear scale, 3 = Real EM Gain (number = multiplication factor).
        """
        if mode not in (0, 1, 2, 3):
            return 20066  # DRV_P1INVALID — matches real SDK behavior
        self._state.em_gain_mode = mode
        return DRV_SUCCESS

    def GetEMCCDGain(self) -> Tuple[int, int]:
        """Return the current EM gain integer in the active mode's units."""
        return (DRV_SUCCESS, self._state.em_gain)

    def GetTemperature(self) -> Tuple[int, float]:
        """Get current temperature with simulated cooling/warming."""
        with self._lock:
            now = time.time()
            elapsed = now - self._last_temp_check
            self._last_temp_check = now

            # Simulate temperature change (5°C per second cooling, 2°C per second warming)
            if self._state.cooler_on:
                if self._state.temperature > self._state.target_temperature:
                    # Cooling
                    self._state.temperature -= min(5.0 * elapsed, self._state.temperature - self._state.target_temperature)
            else:
                if self._state.temperature < 20.0:
                    # Warming towards room temp
                    self._state.temperature += min(2.0 * elapsed, 20.0 - self._state.temperature)

            # Determine status
            if not self._state.cooler_on:
                return (DRV_TEMPERATURE_OFF, self._state.temperature)
            elif abs(self._state.temperature - self._state.target_temperature) < 1.0:
                return (DRV_TEMPERATURE_STABILIZED, self._state.temperature)
            else:
                return (DRV_TEMPERATURE_NOT_REACHED, self._state.temperature)

    def SetTemperature(self, target: int) -> int:
        """Set target temperature."""
        self._state.target_temperature = target
        return DRV_SUCCESS

    def CoolerON(self) -> int:
        """Turn cooler on."""
        self._state.cooler_on = True
        return DRV_SUCCESS

    def CoolerOFF(self) -> int:
        """Turn cooler off."""
        self._state.cooler_on = False
        return DRV_SUCCESS

    def SetExposureTime(self, exposure: float) -> int:
        """Set exposure time in seconds."""
        self._state.exposure_time = exposure
        return DRV_SUCCESS

    def SetAcquisitionMode(self, mode: int) -> int:
        """Set acquisition mode."""
        self._state.acquisition_mode = mode
        return DRV_SUCCESS

    def SetReadMode(self, mode: int) -> int:
        """Set read mode."""
        self._state.read_mode = mode
        return DRV_SUCCESS

    def SetTriggerMode(self, mode: int) -> int:
        """Set trigger mode (0=internal, 1=external)."""
        self._state.trigger_mode = mode
        return DRV_SUCCESS

    def SetFastExtTrigger(self, mode: int) -> int:
        """Enable (1) or disable (0) fast external trigger.

        Only meaningful when trigger mode is external (1).
        """
        self._state.fast_ext_trigger = mode
        return DRV_SUCCESS

    def SetImage(
        self, hbin: int, vbin: int, hstart: int, hend: int, vstart: int, vend: int
    ) -> int:
        """Set image parameters (for Image mode)."""
        self._state.hbin = hbin
        self._state.vbin = vbin
        return DRV_SUCCESS

    def SetFVBHBin(self, hbin: int) -> int:
        """Set horizontal binning for FVB mode.

        IMPORTANT: SetImage is IGNORED in FVB mode. Use this method instead.
        """
        self._state.hbin = hbin
        return DRV_SUCCESS

    # -- Camera settings: VS/HS speed, amplifier, gain ------------------------

    # DU970P VS speeds (µs/pixel): index → speed
    _VS_SPEEDS = [4.9, 9.8, 19.0, 38.0, 57.0]
    # DU970P HS speeds (MHz): index → speed (same for both amplifier types)
    _HS_SPEEDS = [3.0, 1.0, 0.05]
    # Pre-amp gain options: index → gain factor
    _PREAMP_GAINS = [1.0, 2.0, 4.0]

    def GetNumberVSSpeeds(self) -> Tuple[int, int]:
        """Get number of vertical shift speeds."""
        return (DRV_SUCCESS, len(self._VS_SPEEDS))

    def GetVSSpeed(self, index: int) -> Tuple[int, float]:
        """Get vertical shift speed in µs/pixel for given index."""
        return (DRV_SUCCESS, self._VS_SPEEDS[index])

    def SetVSSpeed(self, index: int) -> int:
        """Set vertical shift speed by index."""
        self._state.vs_speed_index = index
        return DRV_SUCCESS

    def GetFastestRecommendedVSSpeed(self) -> Tuple[int, int, float]:
        """Get fastest recommended VS speed index and value."""
        return (DRV_SUCCESS, 0, self._VS_SPEEDS[0])

    def GetNumberHSSpeeds(self, channel: int, typ: int) -> Tuple[int, int]:
        """Get number of horizontal shift speeds."""
        return (DRV_SUCCESS, len(self._HS_SPEEDS))

    def GetHSSpeed(self, channel: int, typ: int, index: int) -> Tuple[int, float]:
        """Get horizontal shift speed in MHz for given index."""
        return (DRV_SUCCESS, self._HS_SPEEDS[index])

    def SetHSSpeed(self, typ: int, index: int) -> int:
        """Set horizontal shift speed."""
        self._state.amplifier_type = typ
        self._state.hs_speed_index = index
        return DRV_SUCCESS

    def GetReadOutTime(self) -> Tuple[int, float]:
        """Return estimated readout time in seconds based on current state.

        Uses the same formula as the readout time calculator but avoids
        importing from andor_qt to prevent circular dependencies.
        """
        _VS_US = {0: 4.9, 1: 9.8, 2: 19.0, 3: 38.0, 4: 57.0}
        _HS_HZ = {0: 3_000_000.0, 1: 1_000_000.0, 2: 50_000.0}

        vs_us = _VS_US.get(self._state.vs_speed_index, 9.8)
        hs_hz = _HS_HZ.get(self._state.hs_speed_index, 1_000_000.0)
        hbin = max(1, self._state.hbin)

        if self._state.crop_mode_active:
            n_rows = self._state.crop_height
            n_px = self._state.crop_width
        else:
            n_rows = self._state.ypixels
            n_px = self._state.xpixels

        # Crop / FVB: all rows shift in, then one horizontal readout
        eff_px = max(1, n_px // hbin)
        t_s = (n_rows * vs_us / 1_000_000.0) + (eff_px / hs_hz)
        return DRV_SUCCESS, t_s

    def SetOutputAmplifier(self, typ: int) -> int:
        """Set output amplifier: 0=EM, 1=conventional."""
        self._state.amplifier_type = typ
        return DRV_SUCCESS

    def SetEMCCDGain(self, gain: int) -> int:
        """Set EM gain."""
        self._state.em_gain = gain
        return DRV_SUCCESS

    def GetNumberPreAmpGains(self) -> Tuple[int, int]:
        """Get number of pre-amplifier gain settings."""
        return (DRV_SUCCESS, len(self._PREAMP_GAINS))

    def GetPreAmpGain(self, index: int) -> Tuple[int, float]:
        """Get pre-amplifier gain for given index."""
        return (DRV_SUCCESS, self._PREAMP_GAINS[index])

    def SetPreAmpGain(self, index: int) -> int:
        """Set pre-amplifier gain by index."""
        self._state.preamp_gain_index = index
        return DRV_SUCCESS

    def SetSingleTrack(self, centre: int, height: int) -> int:
        """Set single track readout mode parameters."""
        self._state.single_track_centre = centre
        self._state.single_track_height = height
        return DRV_SUCCESS

    def SetIsolatedCropMode(
        self, active: int, cropheight: int, cropwidth: int, vbin: int, hbin: int
    ) -> int:
        """Set isolated crop mode."""
        self._state.crop_mode_active = bool(active)
        self._state.crop_height = cropheight
        self._state.crop_width = cropwidth
        self._state.vbin = vbin
        self._state.hbin = hbin
        return DRV_SUCCESS

    def SetOverlapMode(self, mode: int) -> int:
        """Enable or disable overlap mode."""
        return DRV_SUCCESS

    def EnableKeepCleans(self, mode: int) -> int:
        """Enable (1) or disable (0) keep-clean cycles."""
        self._state.keep_cleans = mode
        return DRV_SUCCESS

    def SetBaselineClamp(self, state: int) -> int:
        """Enable or disable baseline clamp."""
        return DRV_SUCCESS

    def GetBaselineClamp(self) -> Tuple[int, int]:
        """Return baseline clamp state."""
        return (DRV_SUCCESS, 1)

    def GetStatus(self) -> Tuple[int, int]:
        """Return camera status (DRV_IDLE=20073 or DRV_ACQUIRING=20072)."""
        if self._state.acquiring:
            return (DRV_SUCCESS, DRV_ACQUIRING)
        return (DRV_SUCCESS, 20073)  # DRV_IDLE

    def PrepareAcquisition(self) -> int:
        """Prepare for acquisition."""
        return DRV_SUCCESS

    def StartAcquisition(self) -> int:
        """Start acquisition."""
        self._state.acquiring = True
        return DRV_SUCCESS

    def AbortAcquisition(self) -> int:
        """Abort acquisition."""
        self._state.acquiring = False
        return DRV_SUCCESS

    def WaitForAcquisition(self) -> int:
        """Wait for acquisition to complete."""
        # Simulate exposure time (capped at 10 ms so tests stay fast)
        time.sleep(min(self._state.exposure_time, 0.010))
        self._state.acquiring = False
        return DRV_SUCCESS

    def GetImages16(
        self, first: int, last: int, size: int
    ) -> Tuple[int, List[int], int, int]:
        """Get acquired image data as 16-bit."""
        n_frames = last - first + 1

        if self._state.read_mode == 0:  # FVB
            eff_x = self._state.xpixels // self._state.hbin
            # Bulk read: concatenate n_frames spectra, each in readout order
            spectra = [self._readout_order(self._generate_mock_spectrum(eff_x))
                       for _ in range(n_frames)]
            data = np.concatenate(spectra)
        else:  # Image
            eff_x = self._state.xpixels // self._state.hbin
            eff_y = self._state.ypixels // self._state.vbin
            data = self._readout_order(self._generate_mock_image(eff_x, eff_y)).flatten()

        return (DRV_SUCCESS, data.tolist(), first, last)

    def IsReadoutFlippedByAmplifier(self, amplifier: int) -> Tuple[int, int]:
        """DU970P: the EM register (0) reads right-to-left, conventional (1) left-to-right."""
        return (DRV_SUCCESS, 1 if amplifier == 0 else 0)

    def _readout_order(self, data: np.ndarray) -> np.ndarray:
        """Return data as the selected amplifier reads it out (columns mirrored on EM)."""
        if self.IsReadoutFlippedByAmplifier(self._state.amplifier_type)[1]:
            return data[..., ::-1]
        return data

    def GetSizeOfCircularBuffer(self) -> Tuple[int, int]:
        """Return maximum circular buffer capacity."""
        return (DRV_SUCCESS, 15000)

    def GetNumberNewImages(self) -> Tuple[int, int, int]:
        """Return the range of new images in the circular buffer."""
        # Mock: pretend we have some frames available
        n = getattr(self._state, "_mock_rta_frames", 100)
        return (DRV_SUCCESS, 1, n)

    def GetOldestImage16(self, size: int) -> Tuple[int, List[int]]:
        """Get oldest image from circular buffer as 16-bit."""
        eff_x = self._state.xpixels // self._state.hbin
        data = self._readout_order(self._generate_mock_spectrum(eff_x))
        return (DRV_SUCCESS, data.tolist())

    def _generate_mock_spectrum(self, xpixels: int) -> np.ndarray:
        """Generate mock pump-probe spectrum alternating pump-on / pump-off.

        Odd shots are pump-on with a delay-dependent ΔI/I₀ response:
          - GSB  (+): Gaussian bleach at centre pixel
          - SE   (+): Gaussian stimulated emission, red-shifted
          - ESA  (−): Gaussian excited-state absorption, blue-shifted
          - Amplitude decays as exp(−t/τ) and carries a coherent oscillation
            exp(−t/τ_vib) · cos(2π t / T_osc).
        """
        x = np.arange(xpixels)
        cx = xpixels // 2  # centre pixel

        # --- Reference (pump-off) spectrum ---
        ref = np.zeros(xpixels)
        for center, width, height in [
            (cx - xpixels // 6, xpixels // 30, 8000),
            (cx, xpixels // 20, 15000),
            (cx + xpixels // 6, xpixels // 25, 10000),
        ]:
            ref += height * np.exp(-0.5 * ((x - center) / width) ** 2)
        ref += np.random.normal(100, 15, xpixels)

        is_pumped = (self._shot_counter % 2 == 1)
        self._shot_counter += 1

        if not is_pumped:
            return np.maximum(ref, 0).astype(np.int32)

        # --- Delay-dependent pump-on perturbation ---
        t = _mock_delay_ps        # current delay set by mock stage
        t0 = 0.0                  # time zero
        tau = 100.0               # population lifetime (ps)
        tau_vib = 3.0             # vibrational dephasing (ps)
        T_osc = 1.5               # oscillation period (ps)

        if t <= t0:
            amp = 0.0
        else:
            dt = t - t0
            amp = np.exp(-dt / tau) * (
                1.0 + 0.25 * np.exp(-dt / tau_vib) * np.cos(2 * np.pi * dt / T_osc)
            )

        w = xpixels // 20
        delta = (
            +1500 * amp * np.exp(-0.5 * ((x - cx) / w) ** 2)               # GSB
            + 800 * amp * np.exp(-0.5 * ((x - cx - w * 2) / (w * 1.5)) ** 2)  # SE
            - 600 * amp * np.exp(-0.5 * ((x - cx + w * 2) / (w * 1.2)) ** 2)  # ESA
        )

        return np.maximum(ref + delta, 0).astype(np.int32)

    def _generate_mock_image(self, xpixels: int, ypixels: int) -> np.ndarray:
        """Generate mock 2D image."""
        y = np.arange(ypixels)
        x = np.arange(xpixels)
        Y, X = np.meshgrid(y, x, indexing='ij')

        # Create image with Gaussian blob
        cx, cy = xpixels // 2, ypixels // 2
        sigma = min(xpixels, ypixels) // 4
        image = 10000 * np.exp(-((X - cx)**2 + (Y - cy)**2) / (2 * sigma**2))

        # Add noise
        image += np.random.normal(100, 20, image.shape)
        return np.maximum(image, 0).astype(np.int32)


class MockATSpectrograph:
    """Mock ATSpectrograph SDK for spectrograph control.

    This class simulates the Andor Spectrograph SDK for testing.
    """

    ATSPECTROGRAPH_SUCCESS = ATSPECTROGRAPH_SUCCESS

    def __init__(self, dll_path: str = ""):
        self._state = MockSpectrographState()
        self._dll_path = dll_path

    def Initialize(self, path: str) -> int:
        """Initialize the spectrograph SDK."""
        self._state.initialized = True
        return ATSPECTROGRAPH_SUCCESS

    def Close(self) -> int:
        """Close the spectrograph SDK."""
        self._state.initialized = False
        return ATSPECTROGRAPH_SUCCESS

    def GetNumberDevices(self) -> Tuple[int, int]:
        """Get number of connected spectrographs."""
        return (ATSPECTROGRAPH_SUCCESS, self._state.num_devices)

    def GetSerialNumber(self, device: int, buf_len: int) -> Tuple[int, str]:
        """Get spectrograph serial number."""
        return (ATSPECTROGRAPH_SUCCESS, self._state.serial_number)

    def GetFunctionReturnDescription(
        self, ret_code: int, buf_len: int
    ) -> Tuple[int, str]:
        """Get error description."""
        if ret_code == ATSPECTROGRAPH_SUCCESS:
            return (ATSPECTROGRAPH_SUCCESS, "Success")
        return (ATSPECTROGRAPH_SUCCESS, f"Error code: {ret_code}")

    def IsGratingPresent(self, device: int) -> Tuple[int, bool]:
        """Check if grating turret is present."""
        return (ATSPECTROGRAPH_SUCCESS, True)

    def IsShutterPresent(self, device: int) -> Tuple[int, bool]:
        """Check if shutter is present."""
        return (ATSPECTROGRAPH_SUCCESS, True)

    def IsFilterPresent(self, device: int) -> Tuple[int, bool]:
        """Check if filter wheel is present."""
        return (ATSPECTROGRAPH_SUCCESS, False)

    def GetNumberGratings(self, device: int) -> Tuple[int, int]:
        """Get number of gratings installed."""
        return (ATSPECTROGRAPH_SUCCESS, self._state.num_gratings)

    def GetGratingInfo(
        self, device: int, grating: int, buf_len: int
    ) -> Tuple[int, float, str, int, int]:
        """Get grating info."""
        if 1 <= grating <= len(self._state.gratings):
            g = self._state.gratings[grating - 1]
            return (ATSPECTROGRAPH_SUCCESS, g["lines"], g["blaze"], 0, 0)
        return (1, 0.0, "", 0, 0)  # Error

    def GetWavelengthLimits(
        self, device: int, grating: int
    ) -> Tuple[int, float, float]:
        """Get wavelength limits for a grating."""
        if 1 <= grating <= len(self._state.gratings):
            g = self._state.gratings[grating - 1]
            return (ATSPECTROGRAPH_SUCCESS, g["wl_min"], g["wl_max"])
        return (1, 0.0, 0.0)  # Error

    def GetGrating(self, device: int) -> Tuple[int, int]:
        """Get current grating."""
        return (ATSPECTROGRAPH_SUCCESS, self._state.current_grating)

    def SetGrating(self, device: int, grating: int) -> int:
        """Set grating (blocking call)."""
        if 1 <= grating <= self._state.num_gratings:
            self._state.current_grating = grating
            time.sleep(0.001)  # 1 ms — just enough for async state tests
            return ATSPECTROGRAPH_SUCCESS
        return 1  # Error

    def GetWavelength(self, device: int) -> Tuple[int, float]:
        """Get current wavelength."""
        return (ATSPECTROGRAPH_SUCCESS, self._state.current_wavelength)

    def SetWavelength(self, device: int, wavelength: float) -> int:
        """Set wavelength (blocking call)."""
        grating = self._state.current_grating
        g = self._state.gratings[grating - 1]
        if g["wl_min"] <= wavelength <= g["wl_max"]:
            self._state.current_wavelength = wavelength
            time.sleep(0.001)  # 1 ms — just enough for async state tests
            return ATSPECTROGRAPH_SUCCESS
        return 1  # Error

    def SetNumberPixels(self, device: int, num_pixels: int) -> int:
        """Set number of pixels for calibration."""
        self._state.num_pixels = num_pixels
        return ATSPECTROGRAPH_SUCCESS

    def SetPixelWidth(self, device: int, pixel_width: float) -> int:
        """Set pixel width for calibration."""
        self._state.pixel_width = pixel_width
        return ATSPECTROGRAPH_SUCCESS

    def GetCalibration(
        self, device: int, num_pixels: int
    ) -> Tuple[int, List[float]]:
        """Get wavelength calibration array."""
        center = self._state.current_wavelength
        grating = self._state.current_grating
        g = self._state.gratings[grating - 1]

        # Estimate dispersion based on grating
        nm_per_pixel = 0.1 * (150.0 / g["lines"])
        half_range = num_pixels * nm_per_pixel / 2

        wavelengths = np.linspace(
            center - half_range, center + half_range, num_pixels
        ).tolist()

        return (ATSPECTROGRAPH_SUCCESS, wavelengths)


def create_mock_sdk_modules():
    """Create mock modules that can be patched into sys.modules.

    Returns:
        Dictionary of module name to mock module object.
    """
    # Create mock modules
    mock_atmcd = MagicMock()
    mock_atmcd.atmcd = MockAtmcd
    mock_atmcd.atmcd_codes = MockAtmcdCodes
    mock_atmcd.atmcd_errors = MockAtmcdErrors

    mock_spectrograph = MagicMock()
    mock_spectrograph.spectrograph = MagicMock()
    mock_spectrograph.spectrograph.ATSpectrograph = MockATSpectrograph

    return {
        "pyAndorSDK2": mock_atmcd,
        "pyAndorSDK2.atmcd": mock_atmcd,
        "pyAndorSDK2.atmcd_codes": mock_atmcd.atmcd_codes,
        "pyAndorSDK2.atmcd_errors": mock_atmcd.atmcd_errors,
        "pyAndorSpectrograph": mock_spectrograph,
        "pyAndorSpectrograph.spectrograph": mock_spectrograph.spectrograph,
    }
