"""PyMeasure instrument wrapper for Andor Shamrock spectrograph.

This module wraps the pyAndorSpectrograph library for use with PyMeasure procedures.

Hardware Safety Rules:
- SetGrating() and SetWavelength() are BLOCKING calls - they wait internally
- ALWAYS check wavelength limits before SetWavelength()
- ALWAYS call SetNumberPixels() and SetPixelWidth() before GetCalibration()
"""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())


@dataclass(frozen=True)
class GratingInfo:
    """Information about a spectrograph grating."""

    index: int
    lines_per_mm: float
    blaze: str
    wavelength_min: float = 0.0
    wavelength_max: float = 0.0


@dataclass(frozen=True)
class SpectrographInfo:
    """Immutable spectrograph information."""

    serial_number: str = ""
    num_gratings: int = 0
    gratings: Tuple[GratingInfo, ...] = ()
    has_shutter: bool = False
    has_filter: bool = False


class AndorSpectrograph:
    """PyMeasure instrument wrapper for Andor Shamrock spectrograph.

    This class provides a simplified interface to the Andor spectrograph for use
    in PyMeasure procedures. It handles initialization, grating/wavelength control,
    and calibration with proper safety checks.

    Example:
        spec = AndorSpectrograph()
        spec.initialize()
        spec.grating = 1
        spec.wavelength = 500.0
        wavelengths = spec.get_calibration(1024, 16.0)
        spec.shutdown()
    """

    MOVE_TIMEOUT = 60.0  # seconds

    def __init__(
        self,
        device_index: int = 0,
        sdk_path: str = r"C:\Program Files\Andor SDK",
    ):
        """Initialize the spectrograph wrapper.

        Args:
            device_index: Spectrograph device index (usually 0).
            sdk_path: Path to the Andor SDK installation directory.
        """
        self._device_index = device_index
        self._sdk_path = sdk_path
        self._spc = None
        self._initialized = False
        self._lock = threading.Lock()
        self._info: Optional[SpectrographInfo] = None
        self._calibration_pixels: int = 0
        self._calibration_pixel_width: float = 0.0

    @property
    def info(self) -> Optional[SpectrographInfo]:
        """Get spectrograph information (available after initialization)."""
        return self._info

    @property
    def grating(self) -> int:
        """Get current grating index (1-indexed)."""
        if not self._initialized:
            return 0

        with self._lock:
            ret, grating = self._spc.GetGrating(self._device_index)
            return grating

    @grating.setter
    def grating(self, value: int) -> None:
        """Set grating (1-indexed).

        This is a blocking call - it waits for the grating to finish moving.

        Args:
            value: Grating index (1-indexed).

        Raises:
            RuntimeError: If spectrograph not initialized or move fails.
            ValueError: If grating index is invalid.
        """
        if not self._initialized:
            raise RuntimeError("Spectrograph not initialized")

        if self._info and (value < 1 or value > self._info.num_gratings):
            raise ValueError(
                f"Grating {value} out of range (1-{self._info.num_gratings})"
            )

        log.info(f"Setting grating to {value}...")

        with self._lock:
            from pyAndorSpectrograph.spectrograph import ATSpectrograph

            # SetGrating is a BLOCKING call - it waits internally until movement completes
            ret = self._spc.SetGrating(self._device_index, value)
            if ret != ATSpectrograph.ATSPECTROGRAPH_SUCCESS:
                desc = self._spc.GetFunctionReturnDescription(ret, 64)[1]
                raise RuntimeError(f"SetGrating failed: {desc}")

        log.info(f"Grating set to {value}")

    @property
    def wavelength(self) -> float:
        """Get current center wavelength in nm."""
        if not self._initialized:
            return 0.0

        with self._lock:
            ret, wavelength = self._spc.GetWavelength(self._device_index)
            return wavelength

    @wavelength.setter
    def wavelength(self, value: float) -> None:
        """Set center wavelength in nm.

        This is a blocking call - it waits for the wavelength motor to finish.

        Args:
            value: Center wavelength in nm.

        Raises:
            RuntimeError: If spectrograph not initialized or move fails.
            ValueError: If wavelength is out of range for current grating.
        """
        if not self._initialized:
            raise RuntimeError("Spectrograph not initialized")

        # Check limits
        wl_min, wl_max = self.get_wavelength_limits()
        if value < wl_min or value > wl_max:
            raise ValueError(
                f"Wavelength {value}nm out of range [{wl_min}, {wl_max}]nm"
            )

        log.info(f"Setting wavelength to {value}nm...")

        with self._lock:
            from pyAndorSpectrograph.spectrograph import ATSpectrograph

            ret = self._spc.SetWavelength(self._device_index, value)
            if ret != ATSpectrograph.ATSPECTROGRAPH_SUCCESS:
                desc = self._spc.GetFunctionReturnDescription(ret, 64)[1]
                raise RuntimeError(f"SetWavelength failed: {desc}")

        log.info(f"Wavelength set to {value}nm")

    def get_wavelength_limits(self, grating: Optional[int] = None) -> Tuple[float, float]:
        """Get wavelength limits for a grating.

        Args:
            grating: Grating index (1-indexed). Uses current grating if None.

        Returns:
            Tuple of (min_wavelength, max_wavelength) in nm.
        """
        if not self._initialized:
            return (0.0, 0.0)

        if grating is None:
            grating = self.grating

        with self._lock:
            from pyAndorSpectrograph.spectrograph import ATSpectrograph

            ret, wl_min, wl_max = self._spc.GetWavelengthLimits(
                self._device_index, grating
            )
            if ret == ATSpectrograph.ATSPECTROGRAPH_SUCCESS:
                return (wl_min, wl_max)
            return (0.0, 0.0)

    def initialize(self) -> None:
        """Initialize the spectrograph SDK.

        Raises:
            RuntimeError: If initialization fails.
        """
        if self._initialized:
            log.warning("Spectrograph already initialized")
            return

        try:
            from pyAndorSpectrograph.spectrograph import ATSpectrograph

            # The spectrograph DLL is in a specific subdirectory
            dll_path = os.path.join(self._sdk_path, "ATSpectrograph", "x64")
            self._spc = ATSpectrograph(dll_path)

            ret = self._spc.Initialize("")
            if ret != ATSpectrograph.ATSPECTROGRAPH_SUCCESS:
                desc = self._spc.GetFunctionReturnDescription(ret, 64)[1]
                raise RuntimeError(f"Spectrograph initialization failed: {desc}")

            self._initialized = True

            # Check for devices
            ret, num_devices = self._spc.GetNumberDevices()
            if num_devices == 0:
                raise RuntimeError("No spectrograph devices found")

            if self._device_index >= num_devices:
                raise RuntimeError(
                    f"Device index {self._device_index} out of range (0-{num_devices-1})"
                )

            # Get serial number
            ret, serial = self._spc.GetSerialNumber(self._device_index, 64)

            # Check features
            ret, has_grating = self._spc.IsGratingPresent(self._device_index)
            ret, has_shutter = self._spc.IsShutterPresent(self._device_index)
            ret, has_filter = self._spc.IsFilterPresent(self._device_index)

            # Get grating info
            gratings: List[GratingInfo] = []
            num_gratings = 0

            if has_grating:
                ret, num_gratings = self._spc.GetNumberGratings(self._device_index)

                for i in range(1, num_gratings + 1):  # Gratings are 1-indexed
                    ret, lines, blaze, home, offset = self._spc.GetGratingInfo(
                        self._device_index, i, 64
                    )
                    if ret == ATSpectrograph.ATSPECTROGRAPH_SUCCESS:
                        ret, wl_min, wl_max = self._spc.GetWavelengthLimits(
                            self._device_index, i
                        )
                        gratings.append(
                            GratingInfo(
                                index=i,
                                lines_per_mm=lines,
                                blaze=blaze,
                                wavelength_min=wl_min if ret == ATSpectrograph.ATSPECTROGRAPH_SUCCESS else 0,
                                wavelength_max=wl_max if ret == ATSpectrograph.ATSPECTROGRAPH_SUCCESS else 0,
                            )
                        )

            self._info = SpectrographInfo(
                serial_number=serial,
                num_gratings=num_gratings,
                gratings=tuple(gratings),
                has_shutter=bool(has_shutter),
                has_filter=bool(has_filter),
            )

            current_grating = self.grating
            current_wavelength = self.wavelength

            log.info(
                f"Spectrograph initialized: serial={serial}, "
                f"{num_gratings} gratings, current grating={current_grating}, "
                f"wavelength={current_wavelength}nm"
            )

        except ImportError as e:
            raise RuntimeError(f"pyAndorSpectrograph not installed: {e}")

    def get_calibration(self, num_pixels: int, pixel_width: float = 16.0) -> np.ndarray:
        """Get wavelength calibration array.

        Args:
            num_pixels: Number of pixels on the detector.
            pixel_width: Pixel width in micrometers (default 16.0um).

        Returns:
            1D numpy array of wavelengths in nm for each pixel.

        Raises:
            RuntimeError: If calibration fails.
        """
        if not self._initialized:
            raise RuntimeError("Spectrograph not initialized")

        with self._lock:
            from pyAndorSpectrograph.spectrograph import ATSpectrograph

            # Set calibration parameters
            ret = self._spc.SetNumberPixels(self._device_index, num_pixels)
            if ret != ATSpectrograph.ATSPECTROGRAPH_SUCCESS:
                desc = self._spc.GetFunctionReturnDescription(ret, 64)[1]
                raise RuntimeError(f"SetNumberPixels failed: {desc}")

            ret = self._spc.SetPixelWidth(self._device_index, pixel_width)
            if ret != ATSpectrograph.ATSPECTROGRAPH_SUCCESS:
                desc = self._spc.GetFunctionReturnDescription(ret, 64)[1]
                raise RuntimeError(f"SetPixelWidth failed: {desc}")

            # Get calibration
            ret, wavelengths = self._spc.GetCalibration(self._device_index, num_pixels)
            if ret != ATSpectrograph.ATSPECTROGRAPH_SUCCESS:
                desc = self._spc.GetFunctionReturnDescription(ret, 64)[1]
                raise RuntimeError(f"GetCalibration failed: {desc}")

            self._calibration_pixels = num_pixels
            self._calibration_pixel_width = pixel_width

        calibration = np.array(wavelengths)
        log.debug(f"Calibration: {calibration[0]:.2f}nm - {calibration[-1]:.2f}nm")
        return calibration

    def get_binned_calibration(
        self, xpixels: int, pixel_width: float = 16.0, hbin: int = 1
    ) -> np.ndarray:
        """Get the wavelength at the centre of each horizontally binned pixel.

        Do not ask GetCalibration for ``xpixels // hbin`` pixels of width
        ``pixel_width * hbin``: the SDK then labels bin k with unbinned pixel
        ``hbin*k + hbin - 1``, (hbin - 1)/2 px to the red of the bin centre
        (measured with argon on the KY193i, 2026-10-15). Instead the unbinned
        axis is averaged over each bin.

        Args:
            xpixels: Unbinned number of detector pixels.
            pixel_width: Unbinned pixel width in micrometers.
            hbin: Horizontal binning factor (must divide ``xpixels``).

        Returns:
            1D numpy array of ``xpixels // hbin`` wavelengths in nm.
        """
        if hbin < 1 or xpixels % hbin != 0:
            raise ValueError(f"hbin={hbin} must be a positive factor of {xpixels}")
        wavelengths = self.get_calibration(xpixels, pixel_width)
        if hbin == 1:
            return wavelengths
        return wavelengths.reshape(-1, hbin).mean(axis=1)

    def shutdown(self) -> None:
        """Shutdown spectrograph SDK."""
        if not self._initialized:
            return

        try:
            with self._lock:
                self._spc.Close()
                self._initialized = False
                log.info("Spectrograph SDK shutdown complete")

        except Exception as e:
            log.error(f"Error during spectrograph shutdown: {e}")
            self._initialized = False
