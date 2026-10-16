"""Spectrum acquisition procedure with shared hardware support.

This module provides procedures for acquiring spectra that can use
shared hardware instances from the HardwareManager.
"""

from __future__ import annotations

import logging

import numpy as np
from pymeasure.experiment import (
    FloatParameter,
    IntegerParameter,
    Procedure,
)

from andor_qt.procedures.base import SharedHardwareMixin

log = logging.getLogger(__name__)


class SpectrumProcedure(SharedHardwareMixin, Procedure):
    """Single spectrum (FVB) acquisition procedure with shared hardware.

    This procedure acquires a 1D spectrum using Full Vertical Binning (FVB) mode
    and returns wavelength-calibrated data.

    Note: Cooler control is NOT included as a parameter - it should be
    controlled directly through the GUI (immediate action, not queued).
    """

    # Parameters shown in GUI
    exposure_time = FloatParameter(
        "Exposure Time",
        units="s",
        default=0.1,
        minimum=0.001,
        maximum=600.0,
    )
    center_wavelength = FloatParameter(
        "Center Wavelength",
        units="nm",
        default=500.0,
        minimum=0.0,
        maximum=2000.0,
    )
    grating = IntegerParameter(
        "Grating",
        default=1,
        minimum=1,
        maximum=3,
    )
    hbin = IntegerParameter(
        "H Binning",
        default=1,
        minimum=1,
        maximum=16,
    )
    num_accumulations = IntegerParameter(
        "Accumulations",
        default=1,
        minimum=1,
        maximum=1000,
    )
    vs_speed_index = IntegerParameter(
        "VS Speed Index",
        default=1,
        minimum=0,
        maximum=4,
    )
    hs_speed_index = IntegerParameter(
        "HS Speed Index",
        default=0,
        minimum=0,
        maximum=2,
    )
    amplifier_type = IntegerParameter(
        "Amplifier Type",
        default=1,  # Conventional, matching the UI default
        minimum=0,
        maximum=1,
    )
    em_gain = IntegerParameter(
        "EM Gain",
        default=1,
        minimum=1,
        maximum=1000,
    )
    preamp_gain_index = IntegerParameter(
        "Pre-Amp Gain Index",
        default=0,
        minimum=0,
        maximum=10,
    )
    delay_position = FloatParameter(
        "Delay Position",
        units="ps",
        default=0.0,
        minimum=-10000.0,
        maximum=10000.0,
    )

    DATA_COLUMNS = ["Wavelength", "Intensity"]

    def startup(self):
        """Initialize hardware using shared instances if available."""
        log.info("Starting spectrum acquisition procedure...")
        self._init_hardware()

        log.info(f"Camera detector: {self.camera.xpixels}x{self.camera.ypixels}")
        log.info(f"Camera temperature: {self.camera.temperature:.1f}C")
        log.info(f"Spectrograph gratings: {self.spectrograph.info.num_gratings}")

    def execute(self):
        """Run spectrum acquisition."""
        # Set spectrograph grating
        log.info(f"Setting grating to {self.grating}")
        self.spectrograph.grating = self.grating

        if self.should_stop():
            return

        # Set spectrograph wavelength
        log.info(f"Setting center wavelength to {self.center_wavelength}nm")
        self.spectrograph.wavelength = self.center_wavelength

        if self.should_stop():
            return

        # Calculate effective pixels with binning
        eff_xpixels = self.camera.xpixels // self.hbin

        # Get wavelength calibration
        log.info("Getting wavelength calibration...")
        wavelengths = self.spectrograph.get_binned_calibration(
            self.camera.xpixels, self.camera.info.pixel_width, self.hbin
        )
        log.info(f"Wavelength range: {wavelengths[0]:.2f} - {wavelengths[-1]:.2f} nm")

        if self.should_stop():
            return

        # Apply camera settings (amplifier, speeds, gain) before acquisition
        self.camera.apply_camera_settings({
            "vs_speed_index": self.vs_speed_index,
            "hs_speed_index": self.hs_speed_index,
            "amplifier_type": self.amplifier_type,
            "em_gain": self.em_gain,
            "preamp_gain_index": self.preamp_gain_index,
        })

        # Set exposure and acquire
        log.info(f"Acquiring spectrum with {self.exposure_time}s exposure, hbin={self.hbin}...")
        self.camera.set_exposure(self.exposure_time)

        # Accumulate if requested
        if self.num_accumulations > 1:
            accumulated = np.zeros(eff_xpixels, dtype=np.float64)
            for i in range(self.num_accumulations):
                if self.should_stop():
                    return

                spectrum = self.camera.acquire_fvb()
                accumulated += spectrum
                self.emit("progress", 100 * (i + 1) / self.num_accumulations)

            spectrum = accumulated / self.num_accumulations
        else:
            spectrum = self.camera.acquire_fvb()
            self.emit("progress", 100)

        # Emit results
        for wl, intensity in zip(wavelengths, spectrum):
            self.emit("results", {"Wavelength": wl, "Intensity": intensity})

        log.info("Spectrum acquisition complete")

    def shutdown(self):
        """Cleanup hardware (only owned instances)."""
        log.info("Shutting down spectrum procedure...")
        self._cleanup_hardware()
        log.info("Spectrum procedure shutdown complete")


class ImageProcedure(SharedHardwareMixin, Procedure):
    """2D image acquisition procedure with shared hardware.

    This procedure acquires a 2D image from the CCD with optional binning.
    """

    # Parameters shown in GUI
    exposure_time = FloatParameter(
        "Exposure Time",
        units="s",
        default=1.0,
        minimum=0.001,
        maximum=600.0,
    )
    center_wavelength = FloatParameter(
        "Center Wavelength",
        units="nm",
        default=500.0,
        minimum=0.0,
        maximum=2000.0,
    )
    grating = IntegerParameter(
        "Grating",
        default=1,
        minimum=1,
        maximum=3,
    )
    hbin = IntegerParameter(
        "H Binning",
        default=1,
        minimum=1,
        maximum=16,
    )
    vbin = IntegerParameter(
        "V Binning",
        default=1,
        minimum=1,
        maximum=16,
    )
    vs_speed_index = IntegerParameter(
        "VS Speed Index",
        default=1,
        minimum=0,
        maximum=4,
    )
    hs_speed_index = IntegerParameter(
        "HS Speed Index",
        default=0,
        minimum=0,
        maximum=2,
    )
    amplifier_type = IntegerParameter(
        "Amplifier Type",
        default=1,  # Conventional, matching the UI default
        minimum=0,
        maximum=1,
    )
    em_gain = IntegerParameter(
        "EM Gain",
        default=1,
        minimum=1,
        maximum=1000,
    )
    preamp_gain_index = IntegerParameter(
        "Pre-Amp Gain Index",
        default=0,
        minimum=0,
        maximum=10,
    )
    delay_position = FloatParameter(
        "Delay Position",
        units="ps",
        default=0.0,
        minimum=-10000.0,
        maximum=10000.0,
    )

    DATA_COLUMNS = ["Wavelength", "Y_Position", "Intensity"]

    def startup(self):
        """Initialize hardware using shared instances if available."""
        log.info("Starting image acquisition procedure...")
        self._init_hardware()

        log.info(f"Camera detector: {self.camera.xpixels}x{self.camera.ypixels}")
        log.info(f"Camera temperature: {self.camera.temperature:.1f}C")

    def execute(self):
        """Run 2D image acquisition."""
        # Set spectrograph grating
        log.info(f"Setting grating to {self.grating}")
        self.spectrograph.grating = self.grating

        if self.should_stop():
            return

        # Set spectrograph wavelength
        log.info(f"Setting center wavelength to {self.center_wavelength}nm")
        self.spectrograph.wavelength = self.center_wavelength

        if self.should_stop():
            return

        # Get wavelength calibration
        eff_xpixels = self.camera.xpixels // self.hbin
        log.info("Getting wavelength calibration...")
        wavelengths = self.spectrograph.get_binned_calibration(
            self.camera.xpixels, self.camera.info.pixel_width, self.hbin
        )
        log.info(f"Wavelength range: {wavelengths[0]:.2f} - {wavelengths[-1]:.2f} nm")

        if self.should_stop():
            return

        # Calculate effective dimensions
        eff_ypixels = self.camera.ypixels // self.vbin

        # Apply camera settings (amplifier, speeds, gain) before acquisition
        self.camera.apply_camera_settings({
            "vs_speed_index": self.vs_speed_index,
            "hs_speed_index": self.hs_speed_index,
            "amplifier_type": self.amplifier_type,
            "em_gain": self.em_gain,
            "preamp_gain_index": self.preamp_gain_index,
        })

        # Set exposure and acquire
        log.info(
            f"Acquiring {eff_xpixels}x{eff_ypixels} image with {self.exposure_time}s exposure..."
        )
        self.camera.set_exposure(self.exposure_time)
        image = self.camera.acquire_image(hbin=self.hbin, vbin=self.vbin)

        self.emit("progress", 50)

        # Emit results (flatten 2D to individual data points)
        log.info("Emitting image data...")
        image.shape[0] * image.shape[1]
        points_emitted = 0

        for y_idx in range(image.shape[0]):
            if self.should_stop():
                return

            for x_idx in range(image.shape[1]):
                self.emit(
                    "results",
                    {
                        "Wavelength": wavelengths[x_idx],
                        "Y_Position": y_idx,
                        "Intensity": image[y_idx, x_idx],
                    },
                )
                points_emitted += 1

            # Update progress
            self.emit("progress", 50 + 50 * (y_idx + 1) / image.shape[0])

        log.info(f"Image acquisition complete: {points_emitted} data points")

    def shutdown(self):
        """Cleanup hardware (only owned instances)."""
        log.info("Shutting down image procedure...")
        self._cleanup_hardware()
        log.info("Image procedure shutdown complete")
