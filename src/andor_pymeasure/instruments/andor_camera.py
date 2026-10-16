"""PyMeasure instrument wrapper for Andor SDK2 camera.

This module wraps the pyAndorSDK2 library for use with PyMeasure procedures.

Hardware Safety Rules:
- ALWAYS call CoolerON() before acquisition for optimal noise
- ALWAYS warm up camera (temp > -20C) before shutdown
- NEVER call Initialize() if already initialized
- Use locks for thread safety during SDK calls
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())


@dataclass(frozen=True)
class CameraInfo:
    """Immutable camera information."""

    serial_number: str = ""
    model: str = ""
    xpixels: int = 0
    ypixels: int = 0
    pixel_width: float = 0.0
    pixel_height: float = 0.0
    em_gain_range: Tuple[int, int] = (0, 0)


class AndorCamera:
    """PyMeasure instrument wrapper for Andor SDK2 camera.

    This class provides a simplified interface to the Andor camera for use
    in PyMeasure procedures. It handles initialization, acquisition, and
    shutdown with proper safety checks.

    Example:
        camera = AndorCamera()
        camera.initialize()
        camera.set_exposure(0.1)
        camera.cooler_on(target=-60)
        data = camera.acquire_fvb()
        camera.shutdown()
    """

    def __init__(self, sdk_path: str = r"C:\Program Files\Andor SDK"):
        """Initialize the camera wrapper.

        Args:
            sdk_path: Path to the Andor SDK installation directory.
        """
        self._sdk_path = sdk_path
        self._sdk = None
        self._codes = None
        self._errors = None
        self._initialized = False
        self._lock = threading.Lock()
        self._info: Optional[CameraInfo] = None
        self._cooler_on = False
        self._current_hbin: int = 1  # tracks last applied hbin for get_spectrum()
        self._em_gain_mode: int = 0  # set by initialize(); 3 = "Real EM Gain"
        # True when the active output amplifier reads the CCD right-to-left
        # (EM register on the DU970P); data are then reversed so that pixel 0
        # is always the short-wavelength end, matching GetCalibration.
        self._readout_flipped: bool = False

    @property
    def info(self) -> Optional[CameraInfo]:
        """Get camera information (available after initialization)."""
        return self._info

    @property
    def readout_flipped(self) -> bool:
        """Whether the active amplifier reads right-to-left (data are corrected)."""
        return self._readout_flipped

    def _refresh_readout_flip(self, amplifier_type: int) -> None:
        """Query the readout direction of ``amplifier_type`` (caller holds the lock)."""
        ret, flipped = self._sdk.IsReadoutFlippedByAmplifier(amplifier_type)
        if ret != self._errors.Error_Codes.DRV_SUCCESS:
            log.warning(
                f"IsReadoutFlippedByAmplifier({amplifier_type}) returned {ret}; "
                "assuming left-to-right readout"
            )
            flipped = 0
        self._readout_flipped = bool(flipped)

    def _to_short_wavelength_first(self, data: np.ndarray) -> np.ndarray:
        """Reverse the column axis when the active amplifier reads right-to-left."""
        if self._readout_flipped:
            return np.ascontiguousarray(data[..., ::-1])
        return data

    @property
    def xpixels(self) -> int:
        """Get horizontal pixel count."""
        return self._info.xpixels if self._info else 0

    @property
    def ypixels(self) -> int:
        """Get vertical pixel count."""
        return self._info.ypixels if self._info else 0

    @property
    def em_gain_mode(self) -> int:
        """Get the active EM gain interpretation mode.

        Returns:
            0 = 8-bit DAC (default at SDK boot, opaque code 0-255),
            1 = 12-bit DAC, 2 = Linear, 3 = Real EM Gain (integer = actual
            electron multiplication factor).
        """
        return self._em_gain_mode

    @property
    def temperature(self) -> float:
        """Get current temperature in Celsius."""
        if not self._initialized:
            return 20.0

        with self._lock:
            ret, temp = self._sdk.GetTemperature()
            return float(temp)

    @property
    def _acq_ok_codes(self) -> set:
        """Return codes that indicate a successful acquisition result.

        The Andor SDK may return temperature status codes (e.g. DRV_TEMPERATURE_STABILIZED)
        from acquisition functions when the cooler reaches its target mid-acquisition.
        These are informational — the image data is still valid.
        """
        ec = self._errors.Error_Codes
        return {
            ec.DRV_SUCCESS,
            ec.DRV_TEMPERATURE_STABILIZED,
            ec.DRV_TEMPERATURE_NOT_REACHED,
            ec.DRV_TEMPERATURE_DRIFT,
            ec.DRV_TEMPERATURE_NOT_STABILIZED,
        }

    @property
    def temperature_status(self) -> str:
        """Get temperature status string."""
        if not self._initialized:
            return "NOT_INITIALIZED"

        with self._lock:
            ret, _ = self._sdk.GetTemperature()
            status_map = {
                self._errors.Error_Codes.DRV_TEMPERATURE_STABILIZED: "STABILIZED",
                self._errors.Error_Codes.DRV_TEMPERATURE_NOT_REACHED: "NOT_REACHED",
                self._errors.Error_Codes.DRV_TEMPERATURE_DRIFT: "DRIFTING",
                self._errors.Error_Codes.DRV_TEMPERATURE_NOT_STABILIZED: "NOT_STABILIZED",
                self._errors.Error_Codes.DRV_TEMPERATURE_OFF: "OFF",
            }
            return status_map.get(ret, "UNKNOWN")

    def initialize(self) -> None:
        """Initialize the camera SDK.

        Raises:
            RuntimeError: If initialization fails.
        """
        if self._initialized:
            log.warning("Camera already initialized")
            return

        try:
            from pyAndorSDK2 import atmcd, atmcd_codes, atmcd_errors

            self._sdk = atmcd(self._sdk_path)
            self._codes = atmcd_codes
            self._errors = atmcd_errors

            ret = self._sdk.Initialize("")
            if ret != self._errors.Error_Codes.DRV_SUCCESS:
                raise RuntimeError(f"Camera initialization failed with code: {ret}")

            self._initialized = True

            # Get camera info
            ret, serial = self._sdk.GetCameraSerialNumber()
            ret, xpixels, ypixels = self._sdk.GetDetector()
            ret, xsize, ysize = self._sdk.GetPixelSize()

            # Switch EM gain to Real EM Gain (mode 3) so the integer passed
            # to SetEMCCDGain is the actual electron multiplication factor
            # rather than an opaque 8-bit DAC code. Must come BEFORE
            # GetEMGainRange so the recorded range reflects physical gains.
            # Cameras without EM amp / older firmware may return non-success
            # — fall back to the SDK default (mode 0) in that case.
            ret_em = self._sdk.SetEMGainMode(3)
            if ret_em == self._errors.Error_Codes.DRV_SUCCESS:
                self._em_gain_mode = 3
            else:
                self._em_gain_mode = 0
                log.warning(
                    f"SetEMGainMode(3) returned {ret_em} — falling back to "
                    "default DAC mode; em_gain values will be opaque codes"
                )

            ret, em_low, em_high = self._sdk.GetEMGainRange()

            self._info = CameraInfo(
                serial_number=str(serial),
                model="Andor Camera",
                xpixels=xpixels,
                ypixels=ypixels,
                pixel_width=xsize,
                pixel_height=ysize,
                em_gain_range=(em_low, em_high),
            )

            # The SDK's default output amplifier after Initialize is 0 (EM)
            self._refresh_readout_flip(0)

            # Set default modes
            self._sdk.SetAcquisitionMode(self._codes.Acquisition_Mode.SINGLE_SCAN)
            self._sdk.SetReadMode(self._codes.Read_Mode.FULL_VERTICAL_BINNING)
            self._sdk.SetTriggerMode(self._codes.Trigger_Mode.INTERNAL)

            log.info(
                f"Camera initialized: serial={serial}, detector={xpixels}x{ypixels}"
            )

        except ImportError as e:
            raise RuntimeError(f"pyAndorSDK2 not installed: {e}")

    def set_exposure(self, exposure_time: float) -> None:
        """Set exposure time in seconds.

        Args:
            exposure_time: Exposure time in seconds.

        Raises:
            RuntimeError: If camera not initialized.
        """
        if not self._initialized:
            raise RuntimeError("Camera not initialized")

        with self._lock:
            self._sdk.SetExposureTime(exposure_time)
            log.debug(f"Exposure time set to {exposure_time}s")

    def cooler_on(self, target: int = -60) -> None:
        """Turn on the cooler with target temperature.

        Args:
            target: Target temperature in Celsius (default -60C).

        Raises:
            RuntimeError: If camera not initialized.
        """
        if not self._initialized:
            raise RuntimeError("Camera not initialized")

        with self._lock:
            self._sdk.SetTemperature(target)
            self._sdk.CoolerON()
            self._cooler_on = True
            log.info(f"Cooler ON, target temperature: {target}C")

    def cooler_off(self) -> None:
        """Turn off the cooler.

        Raises:
            RuntimeError: If camera not initialized.
        """
        if not self._initialized:
            raise RuntimeError("Camera not initialized")

        with self._lock:
            self._sdk.CoolerOFF()
            self._cooler_on = False
            log.info("Cooler OFF")

    def acquire_fvb(self, hbin: int = 1) -> np.ndarray:
        """Acquire single FVB (Full Vertical Binning) spectrum.

        Args:
            hbin: Horizontal binning factor. Must be a factor of xpixels.

        Returns:
            1D numpy array of intensities.

        Raises:
            RuntimeError: If acquisition fails.
            ValueError: If hbin is invalid.
        """
        if not self._initialized:
            raise RuntimeError("Camera not initialized")

        xpixels = self._info.xpixels

        if hbin < 1:
            raise ValueError(f"hbin must be >= 1, got {hbin}")
        if xpixels % hbin != 0:
            raise ValueError(f"hbin={hbin} must be a factor of {xpixels}")

        eff_pixels = xpixels // hbin

        with self._lock:
            # Disable crop mode if it was previously active
            self._sdk.SetIsolatedCropMode(0, 200, self._info.xpixels, 1, 1)

            # Set FVB mode
            self._sdk.SetReadMode(self._codes.Read_Mode.FULL_VERTICAL_BINNING)
            self._sdk.SetAcquisitionMode(self._codes.Acquisition_Mode.SINGLE_SCAN)

            # IMPORTANT: Use SetFVBHBin for horizontal binning in FVB mode.
            # SetImage is IGNORED in FVB mode!
            if hbin > 1:
                ret = self._sdk.SetFVBHBin(hbin)
                if ret != self._errors.Error_Codes.DRV_SUCCESS:
                    log.error(f"SetFVBHBin({hbin}) failed: {ret}")
                    raise RuntimeError(f"SetFVBHBin failed with code {ret}")
                log.debug(f"FVB mode: {eff_pixels} pixels (hbin={hbin})")
            else:
                # Reset to no binning
                ret = self._sdk.SetFVBHBin(1)
                if ret != self._errors.Error_Codes.DRV_SUCCESS:
                    log.warning(f"SetFVBHBin(1) returned: {ret}")

            # Prepare and start
            ret = self._sdk.PrepareAcquisition()
            if ret != self._errors.Error_Codes.DRV_SUCCESS:
                log.warning(f"PrepareAcquisition returned: {ret}")

            ret = self._sdk.StartAcquisition()
            if ret != self._errors.Error_Codes.DRV_SUCCESS:
                raise RuntimeError(f"StartAcquisition failed with code: {ret}")

        # Wait for acquisition (outside lock so abort can work)
        ret = self._sdk.WaitForAcquisition()
        if ret not in self._acq_ok_codes:
            with self._lock:
                self._sdk.AbortAcquisition()
            raise RuntimeError(f"WaitForAcquisition failed with code: {ret}")

        # Get data - use effective pixels based on binning
        with self._lock:
            ret, arr, validfirst, validlast = self._sdk.GetImages16(
                1, 1, eff_pixels
            )
            if ret not in self._acq_ok_codes:
                raise RuntimeError(f"GetImages16 failed with code: {ret}")

        data = self._to_short_wavelength_first(np.array(arr, dtype=np.float64))
        log.debug(f"FVB acquisition complete: {len(data)} pixels (hbin={hbin})")
        return data


    def get_readout_time(self) -> float:
        """Return SDK-calculated readout time in seconds.

        Call after applying all acquisition settings.  Returns 0.0 if the
        camera is not initialized or the SDK call fails.

        Returns:
            Readout time in seconds.
        """
        if not self._initialized:
            return 0.0
        with self._lock:
            ret, t = self._sdk.GetReadOutTime()
            if ret == self._errors.Error_Codes.DRV_SUCCESS:
                return float(t)
        return 0.0

    def get_spectrum(self) -> np.ndarray:
        """Acquire one FVB spectrum using the current hbin setting.

        Convenience wrapper for TA acquisition loops.  Uses ``_current_hbin``
        set by the last ``apply_camera_settings`` call (default 1).

        Returns:
            1-D numpy array of intensities.
        """
        return self.acquire_fvb(hbin=self._current_hbin)

    def acquire_image(self, hbin: int = 1, vbin: int = 1) -> np.ndarray:
        """Acquire 2D image with optional binning.

        Args:
            hbin: Horizontal binning factor.
            vbin: Vertical binning factor.

        Returns:
            2D numpy array of intensities (rows=y, cols=x).

        Raises:
            RuntimeError: If acquisition fails.
        """
        if not self._initialized:
            raise RuntimeError("Camera not initialized")

        xpixels = self._info.xpixels
        ypixels = self._info.ypixels

        if xpixels % hbin != 0:
            raise ValueError(f"hbin={hbin} must be a factor of {xpixels}")
        if ypixels % vbin != 0:
            raise ValueError(f"vbin={vbin} must be a factor of {ypixels}")

        eff_x = xpixels // hbin
        eff_y = ypixels // vbin
        image_size = eff_x * eff_y

        with self._lock:
            # Set image mode
            self._sdk.SetReadMode(self._codes.Read_Mode.IMAGE)
            self._sdk.SetImage(hbin, vbin, 1, xpixels, 1, ypixels)
            self._sdk.SetAcquisitionMode(self._codes.Acquisition_Mode.SINGLE_SCAN)

            # Prepare and start
            ret = self._sdk.PrepareAcquisition()
            ret = self._sdk.StartAcquisition()
            if ret != self._errors.Error_Codes.DRV_SUCCESS:
                raise RuntimeError(f"StartAcquisition failed with code: {ret}")

        # Wait for acquisition
        ret = self._sdk.WaitForAcquisition()
        if ret not in self._acq_ok_codes:
            with self._lock:
                self._sdk.AbortAcquisition()
            raise RuntimeError(f"WaitForAcquisition failed with code: {ret}")

        # Get data
        with self._lock:
            ret, arr, validfirst, validlast = self._sdk.GetImages16(1, 1, image_size)
            if ret not in self._acq_ok_codes:
                raise RuntimeError(f"GetImages16 failed with code: {ret}")

        data = self._to_short_wavelength_first(
            np.array(arr, dtype=np.float64).reshape(eff_y, eff_x)
        )
        log.debug(f"Image acquisition complete: {eff_y}x{eff_x}")
        return data

    def start_run_till_abort(self) -> None:
        """Start continuous FVB acquisition in Run Till Abort mode.

        The camera arms once and continuously acquires frames on each
        external trigger.  Call ``get_buffered_frames()`` to bulk-read
        accumulated frames, then ``abort_acquisition()`` to stop.
        """
        if not self._initialized:
            raise RuntimeError("Camera not initialized")

        # Ensure no acquisition is running (prevents DRV_ACQUIRING 20072)
        self._ensure_idle()

        hbin = self._current_hbin
        xpixels = self._info.xpixels
        self._rta_eff_pixels = xpixels // hbin

        with self._lock:
            # Disable crop mode if it was previously active
            self._sdk.SetIsolatedCropMode(0, 200, xpixels, 1, 1)

            self._sdk.SetReadMode(self._codes.Read_Mode.FULL_VERTICAL_BINNING)
            self._sdk.SetAcquisitionMode(5)  # RUN_TILL_ABORT

            # Enable overlap mode: readout of frame N overlaps with exposure of frame N+1.
            # Required for 500 Hz frame rate with fast_external trigger.
            ret = self._sdk.SetOverlapMode(1)
            if ret != self._errors.Error_Codes.DRV_SUCCESS:
                log.warning(f"SetOverlapMode(1) returned: {ret}")

            if hbin > 1:
                ret = self._sdk.SetFVBHBin(hbin)
                if ret != self._errors.Error_Codes.DRV_SUCCESS:
                    raise RuntimeError(f"SetFVBHBin failed with code {ret}")
            else:
                self._sdk.SetFVBHBin(1)

            ret = self._sdk.PrepareAcquisition()
            if ret != self._errors.Error_Codes.DRV_SUCCESS:
                log.warning(f"PrepareAcquisition returned: {ret}")

            ret = self._sdk.StartAcquisition()
            if ret != self._errors.Error_Codes.DRV_SUCCESS:
                self._rta_eff_pixels = None
                raise RuntimeError(f"StartAcquisition failed with code: {ret}")

    def start_run_till_abort_crop(
        self, crop_height: int = 50, crop_width: int = 1600,
        vbin: int = 1, hbin: int = 1,
    ) -> None:
        """Start continuous crop-mode acquisition in Run Till Abort mode.

        Uses ``SetIsolatedCropMode`` to restrict the active sensor region,
        reducing readout time to support 1 kHz single-shot triggering.
        FVB mode bins all crop rows into a single row.

        Args:
            crop_height: Number of rows in the crop region (anchored to bottom).
            crop_width: Number of columns (usually 1600 = full width).
            vbin: Vertical binning within the crop region.
            hbin: Horizontal binning factor.
        """
        if not self._initialized:
            raise RuntimeError("Camera not initialized")

        # Ensure no acquisition is running (prevents DRV_ACQUIRING 20072)
        self._ensure_idle()

        self._rta_eff_pixels = crop_width // hbin

        with self._lock:
            # Activate isolated crop mode (region anchored to bottom of sensor)
            log.info(f"Setting crop mode: {crop_height}x{crop_width} vbin={vbin} hbin={hbin}")
            ret = self._sdk.SetIsolatedCropMode(1, crop_height, crop_width, vbin, hbin)
            if ret != self._errors.Error_Codes.DRV_SUCCESS:
                log.error(f"SetIsolatedCropMode failed: {ret}")
                self._rta_eff_pixels = None
                raise RuntimeError(f"SetIsolatedCropMode failed with code: {ret}")

            # FVB bins all crop rows into one row → 1-D spectrum output
            self._sdk.SetReadMode(self._codes.Read_Mode.FULL_VERTICAL_BINNING)
            self._sdk.SetAcquisitionMode(5)  # RUN_TILL_ABORT

            ret = self._sdk.SetOverlapMode(1)
            if ret != self._errors.Error_Codes.DRV_SUCCESS:
                log.warning(f"SetOverlapMode(1) returned: {ret}")

            ret = self._sdk.PrepareAcquisition()
            if ret != self._errors.Error_Codes.DRV_SUCCESS:
                log.warning(f"PrepareAcquisition returned: {ret}")

            ret = self._sdk.StartAcquisition()
            if ret != self._errors.Error_Codes.DRV_SUCCESS:
                log.error(f"StartAcquisition (crop RTA) failed: {ret}")
                self._rta_eff_pixels = None
                raise RuntimeError(f"StartAcquisition failed with code: {ret}")

            log.info(f"Crop RTA started: {crop_height}x{crop_width} eff_pixels={self._rta_eff_pixels}")

    def get_circular_buffer_size(self) -> int:
        """Return the maximum number of frames the circular buffer can hold.

        Must be called after acquisition settings are configured (the buffer
        size depends on frame dimensions and binning).
        """
        if not self._initialized:
            return 0
        with self._lock:
            ret, size = self._sdk.GetSizeOfCircularBuffer()
            if ret != self._errors.Error_Codes.DRV_SUCCESS:
                log.warning(f"GetSizeOfCircularBuffer returned: {ret}")
                return 12000  # conservative fallback
        return int(size)

    def get_buffered_frames(self) -> tuple:
        """Read all available frames from the circular buffer.

        Must be called after ``start_run_till_abort()``.

        Returns:
            ``(frames, n_frames)`` where ``frames`` is a 2-D numpy array
            of shape ``(n_frames, pixels)`` and ``n_frames`` is the count.
        """
        if not self._initialized:
            raise RuntimeError("Camera not initialized")
        eff_pixels = getattr(self, "_rta_eff_pixels", None)
        if eff_pixels is None:
            raise RuntimeError("start_run_till_abort() must be called first")

        with self._lock:
            ret, first, last = self._sdk.GetNumberNewImages()
            if ret != self._errors.Error_Codes.DRV_SUCCESS:
                return np.empty((0, eff_pixels), dtype=np.float64), 0

            n_frames = last - first + 1
            total_pixels = n_frames * eff_pixels
            ret, arr, validfirst, validlast = self._sdk.GetImages16(
                first, last, total_pixels
            )
            if ret not in self._acq_ok_codes:
                raise RuntimeError(f"GetImages16 failed with code: {ret}")

        n_valid = validlast - validfirst + 1
        arr_np = np.array(arr, dtype=np.float64)
        # SDK may return more data than n_valid frames — slice to valid range
        valid_pixels = n_valid * eff_pixels
        frames = self._to_short_wavelength_first(
            arr_np[:valid_pixels].reshape(n_valid, eff_pixels)
        )
        return frames, n_valid

    def _ensure_idle(self) -> None:
        """If camera is acquiring, abort and wait for idle before proceeding.

        Prevents DRV_ACQUIRING (20072) errors when starting a new acquisition
        immediately after the previous one.
        """
        import time
        with self._lock:
            ret, status = self._sdk.GetStatus()
            if status == 20072:  # DRV_ACQUIRING
                log.warning("Camera still acquiring — aborting before restart")
                self._sdk.AbortAcquisition()
                for _ in range(100):
                    ret, status = self._sdk.GetStatus()
                    if status == 20073:  # DRV_IDLE
                        break
                    time.sleep(0.005)

    def abort_acquisition(self) -> None:
        """Abort any running acquisition and wait until camera is idle."""
        if not self._initialized:
            return

        with self._lock:
            self._sdk.AbortAcquisition()
            self._rta_eff_pixels = None

            # Wait for camera to become idle (DRV_IDLE = 20073)
            import time
            for _ in range(100):  # up to 500 ms
                ret, status = self._sdk.GetStatus()
                if status == 20073:  # DRV_IDLE
                    break
                time.sleep(0.005)

            log.info("Acquisition aborted")

    # -- Camera settings: VS/HS speed, amplifier, gain -----------------------

    def get_vs_speeds(self) -> list:
        """Get available vertical shift speeds.

        Returns:
            List of (index, speed_us) tuples.

        Raises:
            RuntimeError: If camera not initialized.
        """
        if not self._initialized:
            raise RuntimeError("Camera not initialized")

        with self._lock:
            _, n = self._sdk.GetNumberVSSpeeds()
            return [(i, self._sdk.GetVSSpeed(i)[1]) for i in range(n)]

    def set_vs_speed(self, index: int) -> None:
        """Set vertical shift speed by index.

        Args:
            index: VS speed index (0=fastest, higher=slower).

        Raises:
            RuntimeError: If camera not initialized.
        """
        if not self._initialized:
            raise RuntimeError("Camera not initialized")

        with self._lock:
            self._sdk.SetVSSpeed(index)
            log.debug(f"VS speed set to index {index}")

    def get_hs_speeds(self, amplifier_type: int = 0) -> list:
        """Get available horizontal shift speeds for an amplifier type.

        Args:
            amplifier_type: 0=EM, 1=conventional.

        Returns:
            List of (index, speed_mhz) tuples.

        Raises:
            RuntimeError: If camera not initialized.
        """
        if not self._initialized:
            raise RuntimeError("Camera not initialized")

        with self._lock:
            _, n = self._sdk.GetNumberHSSpeeds(0, amplifier_type)
            return [(i, self._sdk.GetHSSpeed(0, amplifier_type, i)[1]) for i in range(n)]

    def set_hs_speed(self, amplifier_type: int, index: int) -> None:
        """Set horizontal shift speed.

        Args:
            amplifier_type: 0=EM, 1=conventional.
            index: HS speed index (0=fastest).

        Raises:
            RuntimeError: If camera not initialized.
        """
        if not self._initialized:
            raise RuntimeError("Camera not initialized")

        with self._lock:
            self._sdk.SetHSSpeed(amplifier_type, index)
            log.debug(f"HS speed set to index {index} (amplifier type {amplifier_type})")

    def set_amplifier(self, amplifier_type: int) -> None:
        """Set output amplifier.

        Args:
            amplifier_type: 0=EM (EMCCD), 1=conventional CCD.

        Raises:
            RuntimeError: If camera not initialized.
        """
        if not self._initialized:
            raise RuntimeError("Camera not initialized")

        with self._lock:
            self._sdk.SetOutputAmplifier(amplifier_type)
            self._refresh_readout_flip(amplifier_type)
            log.debug(f"Amplifier set to type {amplifier_type}")

    def set_em_gain(self, gain: int) -> None:
        """Set EM gain.

        Args:
            gain: EM gain value (within range from GetEMGainRange).

        Raises:
            RuntimeError: If camera not initialized.
        """
        if not self._initialized:
            raise RuntimeError("Camera not initialized")

        with self._lock:
            self._sdk.SetEMCCDGain(gain)
            log.debug(f"EM gain set to {gain}")

    def get_em_gain(self) -> int:
        """Return the currently applied EM gain.

        Units depend on ``em_gain_mode``: in mode 3 (Real EM Gain) the value
        is the physical electron multiplication factor; in mode 0/1 it is a
        DAC code.

        Returns:
            Current EM gain integer, or 0 if camera is not initialized / SDK
            call fails.
        """
        if not self._initialized:
            return 0
        with self._lock:
            ret, gain = self._sdk.GetEMCCDGain()
            if ret == self._errors.Error_Codes.DRV_SUCCESS:
                return int(gain)
        return 0

    def get_preamp_gains(self) -> list:
        """Get available pre-amplifier gain options.

        Returns:
            List of (index, gain_factor) tuples.

        Raises:
            RuntimeError: If camera not initialized.
        """
        if not self._initialized:
            raise RuntimeError("Camera not initialized")

        with self._lock:
            _, n = self._sdk.GetNumberPreAmpGains()
            return [(i, self._sdk.GetPreAmpGain(i)[1]) for i in range(n)]

    def set_preamp_gain(self, index: int) -> None:
        """Set pre-amplifier gain by index.

        Args:
            index: Pre-amp gain index.

        Raises:
            RuntimeError: If camera not initialized.
        """
        if not self._initialized:
            raise RuntimeError("Camera not initialized")

        with self._lock:
            self._sdk.SetPreAmpGain(index)
            log.debug(f"Pre-amp gain set to index {index}")

    def set_single_track(self, centre: int, height: int) -> None:
        """Configure single track readout mode.

        Args:
            centre: Centre row of the track (1-indexed from bottom).
            height: Number of rows to bin.

        Raises:
            RuntimeError: If camera not initialized.
        """
        if not self._initialized:
            raise RuntimeError("Camera not initialized")

        with self._lock:
            self._sdk.SetSingleTrack(centre, height)
            log.debug(f"Single track: centre={centre}, height={height}")

    def set_crop_mode(
        self, active: bool, crop_height: int, crop_width: int, vbin: int = 1, hbin: int = 1
    ) -> None:
        """Configure isolated crop mode for fast readout.

        WARNING: Light must not fall on rows outside the cropped region.

        Args:
            active: True to enable crop mode.
            crop_height: Number of active rows.
            crop_width: Number of active columns (typically full width).
            vbin: Vertical binning within the crop.
            hbin: Horizontal binning within the crop.

        Raises:
            RuntimeError: If camera not initialized.
        """
        if not self._initialized:
            raise RuntimeError("Camera not initialized")

        with self._lock:
            self._sdk.SetIsolatedCropMode(int(active), crop_height, crop_width, vbin, hbin)
            log.debug(f"Crop mode: active={active}, {crop_height}x{crop_width}, vbin={vbin}, hbin={hbin}")

    def apply_camera_settings(self, settings: dict) -> None:
        """Apply a camera settings dict before acquisition.

        Applies VS speed, HS speed, amplifier type, EM gain, pre-amp gain,
        and read area mode (full / single_track / crop) from the provided dict.
        Missing keys are silently skipped.

        Args:
            settings: Dict with any of: vs_speed_index, hs_speed_index,
                amplifier_type, em_gain, preamp_gain_index, read_area_mode,
                single_track_centre, single_track_height, crop_height, crop_width.

        Raises:
            RuntimeError: If camera not initialized.
        """
        if not self._initialized:
            raise RuntimeError("Camera not initialized")

        amplifier_type = settings.get("amplifier_type", 0)

        with self._lock:
            if "vs_speed_index" in settings:
                self._sdk.SetVSSpeed(settings["vs_speed_index"])
            if "amplifier_type" in settings:
                self._sdk.SetOutputAmplifier(amplifier_type)
                self._refresh_readout_flip(amplifier_type)
            if "hs_speed_index" in settings:
                self._sdk.SetHSSpeed(amplifier_type, settings["hs_speed_index"])
            if "em_gain" in settings and amplifier_type == 0:
                _, em_low, em_high = self._sdk.GetEMGainRange()
                gain = max(em_low, min(settings["em_gain"], em_high))
                if gain != settings["em_gain"]:
                    log.warning(
                        f"EM gain {settings['em_gain']} clamped to [{em_low}, {em_high}] → {gain}"
                    )
                self._sdk.SetEMCCDGain(gain)
                ret_g, actual = self._sdk.GetEMCCDGain()
                if ret_g == self._errors.Error_Codes.DRV_SUCCESS and actual != gain:
                    log.warning(
                        f"EM gain readback mismatch: set={gain} actual={actual}"
                    )
            if "preamp_gain_index" in settings:
                self._sdk.SetPreAmpGain(settings["preamp_gain_index"])

            hbin = settings.get("hbin", 1)
            vbin = settings.get("vbin", 1)
            self._current_hbin = hbin

            # Read area mode
            mode = settings.get("read_area_mode", "full")
            if mode == "crop":
                crop_h = settings.get("crop_height", 16)
                crop_w = settings.get("crop_width", 1600)
                self._sdk.SetIsolatedCropMode(1, crop_h, crop_w, vbin, hbin)
                log.debug(f"Crop mode enabled: {crop_h}×{crop_w} vbin={vbin} hbin={hbin}")
            elif mode == "single_track":
                centre = settings.get("single_track_centre", 100)
                height_rows = settings.get("single_track_height", 10)
                self._sdk.SetSingleTrack(centre, height_rows)
                log.debug(f"Single track: centre={centre}, height={height_rows}")
            else:
                # Full CCD — ensure crop mode is deactivated
                self._sdk.SetIsolatedCropMode(0, 200, 1600, 1, 1)
                if hbin > 1:
                    self._sdk.SetFVBHBin(hbin)

            # Trigger mode
            if "trigger_mode" in settings:
                trig = settings["trigger_mode"]
                if trig == "fast_external":
                    self._sdk.SetTriggerMode(1)   # external
                    self._sdk.SetFastExtTrigger(1)
                elif trig == "external":
                    self._sdk.SetTriggerMode(1)
                    self._sdk.SetFastExtTrigger(0)
                else:  # internal
                    self._sdk.SetTriggerMode(0)
                    self._sdk.SetFastExtTrigger(0)
                log.debug(f"Trigger mode: {trig}")

            if "exposure_time" in settings:
                self._sdk.SetExposureTime(settings["exposure_time"])
                log.debug(f"Exposure time: {settings['exposure_time']} s")

            if "baseline_clamp" in settings:
                state = 1 if settings["baseline_clamp"] else 0
                ret = self._sdk.SetBaselineClamp(state)
                if ret != self._errors.Error_Codes.DRV_SUCCESS:
                    log.warning(f"SetBaselineClamp({state}) returned: {ret}")
                else:
                    log.debug(f"Baseline clamp: {'ON' if state else 'OFF'}")

            if "keep_cleans" in settings:
                state = 1 if settings["keep_cleans"] else 0
                # EnableKeepCleans requires idle state. Skip silently if acquiring —
                # the setting will take effect on the next idle apply.
                status_ret, status = self._sdk.GetStatus()
                if status == self._errors.Error_Codes.DRV_ACQUIRING:
                    log.debug("Skipping EnableKeepCleans — camera is acquiring")
                else:
                    ret = self._sdk.EnableKeepCleans(state)
                    if ret != self._errors.Error_Codes.DRV_SUCCESS:
                        log.warning(f"EnableKeepCleans({state}) returned: {ret}")
                    else:
                        log.debug(f"Keep cleans: {'ON' if state else 'OFF'}")

        log.debug(f"Camera settings applied: {settings}")

    def warmup(self, target: float = -20.0, timeout: float = 300.0) -> bool:
        """Warm up camera to safe shutdown temperature.

        Args:
            target: Target temperature in Celsius (default -20C).
            timeout: Maximum wait time in seconds (default 300s).

        Returns:
            True if target reached, False if timed out.
        """
        if not self._initialized:
            return True

        # Turn off cooler first
        if self._cooler_on:
            self.cooler_off()

        start = time.time()
        while (time.time() - start) < timeout:
            temp = self.temperature
            log.info(f"Warming up: {temp:.1f}C (target: {target}C)")

            if temp >= target:
                log.info("Warmup complete")
                return True

            time.sleep(5)

        log.warning(f"Warmup timed out after {timeout}s")
        return False

    def shutdown(self) -> None:
        """Shutdown camera SDK.

        IMPORTANT: This method will warm up the camera if temperature is below -20C.
        """
        if not self._initialized:
            return

        try:
            # Safety: warm up before shutdown if cold
            temp = self.temperature
            if temp < -20:
                log.warning(f"Camera is cold ({temp}C). Warming up before shutdown...")
                self.warmup(target=-20, timeout=300)

            with self._lock:
                self._sdk.ShutDown()
                self._initialized = False
                log.info("Camera SDK shutdown complete")

        except Exception as e:
            log.error(f"Error during camera shutdown: {e}")
            # Still try to shutdown even if warmup failed
            try:
                with self._lock:
                    self._sdk.ShutDown()
                    self._initialized = False
            except Exception:
                pass
            raise
