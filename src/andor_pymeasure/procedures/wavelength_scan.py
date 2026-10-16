"""Wavelength scan procedures using PyMeasure.

This module provides procedures for scanning across wavelengths by moving
the spectrograph and acquiring spectra or images at each position.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
from pymeasure.experiment import (
    BooleanParameter,
    FloatParameter,
    IntegerParameter,
    Procedure,
)

if TYPE_CHECKING:
    pass

log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())


class WavelengthScanProcedure(Procedure):
    """Wavelength scan procedure.

    This procedure scans the spectrograph across a wavelength range,
    acquiring a spectrum (FVB mode) at each center wavelength position.
    This allows coverage of a wider spectral range than a single acquisition.
    """

    # Parameters shown in GUI
    wavelength_start = FloatParameter(
        "Start Wavelength",
        units="nm",
        default=400.0,
        minimum=0.0,
        maximum=2000.0,
    )
    wavelength_end = FloatParameter(
        "End Wavelength",
        units="nm",
        default=700.0,
        minimum=0.0,
        maximum=2000.0,
    )
    wavelength_step = FloatParameter(
        "Wavelength Step",
        units="nm",
        default=50.0,
        minimum=1.0,
        maximum=500.0,
    )
    exposure_time = FloatParameter(
        "Exposure Time",
        units="s",
        default=0.1,
        minimum=0.001,
        maximum=600.0,
    )
    grating = IntegerParameter(
        "Grating",
        default=1,
        minimum=1,
        maximum=3,
    )
    cooler_enabled = BooleanParameter(
        "Enable Cooler",
        default=True,
    )
    target_temperature = IntegerParameter(
        "Target Temperature",
        units="C",
        default=-60,
        minimum=-100,
        maximum=20,
    )

    # Each data point includes the center wavelength, pixel wavelength, and intensity
    DATA_COLUMNS = ["Center_Wavelength", "Pixel_Wavelength", "Intensity"]

    def startup(self):
        """Initialize hardware."""
        log.info("Initializing hardware for wavelength scan...")

        from andor_pymeasure.instruments.andor_camera import AndorCamera
        from andor_pymeasure.instruments.andor_spectrograph import AndorSpectrograph

        self.camera = AndorCamera()
        self.camera.initialize()

        self.spectrograph = AndorSpectrograph()
        self.spectrograph.initialize()

        log.info(f"Camera detector: {self.camera.xpixels}x{self.camera.ypixels}")
        log.info(f"Camera temperature: {self.camera.temperature:.1f}C")

        # Enable cooler if requested
        if self.cooler_enabled:
            log.info(f"Enabling cooler, target: {self.target_temperature}C")
            self.camera.cooler_on(target=self.target_temperature)

    def execute(self):
        """Run wavelength scan."""
        # Set grating
        log.info(f"Setting grating to {self.grating}")
        self.spectrograph.grating = self.grating

        if self.should_stop():
            return

        # Check wavelength limits
        wl_min, wl_max = self.spectrograph.get_wavelength_limits()
        log.info(f"Grating wavelength limits: {wl_min:.1f} - {wl_max:.1f} nm")

        # Clip scan range to valid limits
        start = max(self.wavelength_start, wl_min)
        end = min(self.wavelength_end, wl_max)

        if start >= end:
            log.error(f"Invalid wavelength range: {start} - {end}")
            return

        # Generate wavelength positions
        wavelengths = np.arange(start, end + self.wavelength_step, self.wavelength_step)
        num_positions = len(wavelengths)
        log.info(f"Scanning {num_positions} wavelength positions from {start} to {end} nm")

        # Set exposure
        self.camera.set_exposure(self.exposure_time)

        # Scan loop
        for i, center_wl in enumerate(wavelengths):
            if self.should_stop():
                log.warning("Scan aborted by user")
                return

            log.info(f"Position {i+1}/{num_positions}: {center_wl:.1f} nm")

            # Move spectrograph
            self.spectrograph.wavelength = center_wl

            # Get calibration for this position
            pixel_wavelengths = self.spectrograph.get_calibration(
                self.camera.xpixels,
                self.camera.info.pixel_width,
            )

            # Acquire spectrum
            spectrum = self.camera.acquire_fvb()

            # Emit data points
            for pixel_wl, intensity in zip(pixel_wavelengths, spectrum):
                self.emit(
                    "results",
                    {
                        "Center_Wavelength": center_wl,
                        "Pixel_Wavelength": pixel_wl,
                        "Intensity": intensity,
                    },
                )

            # Update progress
            self.emit("progress", 100 * (i + 1) / num_positions)

        log.info("Wavelength scan complete")

    def shutdown(self):
        """Cleanup hardware."""
        log.info("Shutting down hardware...")

        if hasattr(self, "camera"):
            self.camera.shutdown()

        if hasattr(self, "spectrograph"):
            self.spectrograph.shutdown()

        log.info("Hardware shutdown complete")


class WavelengthImageScanProcedure(Procedure):
    """Wavelength scan with 2D image acquisition.

    This procedure scans the spectrograph across a wavelength range,
    acquiring a 2D image at each center wavelength position.
    """

    # Parameters shown in GUI
    wavelength_start = FloatParameter(
        "Start Wavelength",
        units="nm",
        default=400.0,
        minimum=0.0,
        maximum=2000.0,
    )
    wavelength_end = FloatParameter(
        "End Wavelength",
        units="nm",
        default=700.0,
        minimum=0.0,
        maximum=2000.0,
    )
    wavelength_step = FloatParameter(
        "Wavelength Step",
        units="nm",
        default=50.0,
        minimum=1.0,
        maximum=500.0,
    )
    exposure_time = FloatParameter(
        "Exposure Time",
        units="s",
        default=1.0,
        minimum=0.001,
        maximum=600.0,
    )
    grating = IntegerParameter(
        "Grating",
        default=1,
        minimum=1,
        maximum=3,
    )
    hbin = IntegerParameter(
        "Horizontal Binning",
        default=1,
        minimum=1,
        maximum=16,
    )
    vbin = IntegerParameter(
        "Vertical Binning",
        default=1,
        minimum=1,
        maximum=16,
    )
    cooler_enabled = BooleanParameter(
        "Enable Cooler",
        default=True,
    )
    target_temperature = IntegerParameter(
        "Target Temperature",
        units="C",
        default=-60,
        minimum=-100,
        maximum=20,
    )

    DATA_COLUMNS = ["Center_Wavelength", "Pixel_Wavelength", "Y_Position", "Intensity"]

    def startup(self):
        """Initialize hardware."""
        log.info("Initializing hardware for wavelength image scan...")

        from andor_pymeasure.instruments.andor_camera import AndorCamera
        from andor_pymeasure.instruments.andor_spectrograph import AndorSpectrograph

        self.camera = AndorCamera()
        self.camera.initialize()

        self.spectrograph = AndorSpectrograph()
        self.spectrograph.initialize()

        log.info(f"Camera detector: {self.camera.xpixels}x{self.camera.ypixels}")

        # Enable cooler if requested
        if self.cooler_enabled:
            log.info(f"Enabling cooler, target: {self.target_temperature}C")
            self.camera.cooler_on(target=self.target_temperature)

    def execute(self):
        """Run wavelength image scan."""
        # Set grating
        log.info(f"Setting grating to {self.grating}")
        self.spectrograph.grating = self.grating

        if self.should_stop():
            return

        # Check wavelength limits
        wl_min, wl_max = self.spectrograph.get_wavelength_limits()

        # Clip scan range to valid limits
        start = max(self.wavelength_start, wl_min)
        end = min(self.wavelength_end, wl_max)

        if start >= end:
            log.error(f"Invalid wavelength range: {start} - {end}")
            return

        # Generate wavelength positions
        wavelengths = np.arange(start, end + self.wavelength_step, self.wavelength_step)
        num_positions = len(wavelengths)
        log.info(f"Scanning {num_positions} positions with 2D images")

        # Calculate effective dimensions
        self.camera.ypixels // self.vbin

        # Set exposure
        self.camera.set_exposure(self.exposure_time)

        # Scan loop
        for i, center_wl in enumerate(wavelengths):
            if self.should_stop():
                log.warning("Scan aborted by user")
                return

            log.info(f"Position {i+1}/{num_positions}: {center_wl:.1f} nm")

            # Move spectrograph
            self.spectrograph.wavelength = center_wl

            # Get calibration
            pixel_wavelengths = self.spectrograph.get_binned_calibration(
                self.camera.xpixels, self.camera.info.pixel_width, self.hbin
            )

            # Acquire image
            image = self.camera.acquire_image(hbin=self.hbin, vbin=self.vbin)

            # Emit data points
            for y_idx in range(image.shape[0]):
                for x_idx in range(image.shape[1]):
                    self.emit(
                        "results",
                        {
                            "Center_Wavelength": center_wl,
                            "Pixel_Wavelength": pixel_wavelengths[x_idx],
                            "Y_Position": y_idx,
                            "Intensity": image[y_idx, x_idx],
                        },
                    )

            # Update progress
            self.emit("progress", 100 * (i + 1) / num_positions)

        log.info("Wavelength image scan complete")

    def shutdown(self):
        """Cleanup hardware."""
        log.info("Shutting down hardware...")

        if hasattr(self, "camera"):
            self.camera.shutdown()

        if hasattr(self, "spectrograph"):
            self.spectrograph.shutdown()

        log.info("Hardware shutdown complete")
