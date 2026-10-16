"""Pump-probe spectroscopy procedures using PyMeasure.

This module provides procedures for time-resolved pump-probe spectroscopy
experiments that scan a delay stage while acquiring spectra or images.
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


class PumpProbeProcedure(Procedure):
    """Pump-probe spectrum (FVB) procedure.

    This procedure scans a delay stage while acquiring spectra (FVB mode)
    at each delay position. The result is a 2D dataset of intensity vs
    delay time and wavelength.
    """

    # Delay parameters
    delay_start = FloatParameter(
        "Delay Start",
        units="ps",
        default=-10.0,
    )
    delay_end = FloatParameter(
        "Delay End",
        units="ps",
        default=100.0,
    )
    delay_step = FloatParameter(
        "Delay Step",
        units="ps",
        default=1.0,
        minimum=0.001,
    )

    # Spectrograph parameters
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

    # Camera parameters
    exposure_time = FloatParameter(
        "Exposure Time",
        units="s",
        default=0.1,
        minimum=0.001,
        maximum=600.0,
    )
    num_accumulations = IntegerParameter(
        "Accumulations",
        default=1,
        minimum=1,
        maximum=1000,
    )

    # Cooling
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

    # Mock mode for testing
    use_mock_stage = BooleanParameter(
        "Use Mock Stage",
        default=False,
    )

    DATA_COLUMNS = ["Delay", "Wavelength", "Intensity"]

    def startup(self):
        """Initialize hardware."""
        log.info("Initializing hardware for pump-probe experiment...")

        from andor_pymeasure.instruments.andor_camera import AndorCamera
        from andor_pymeasure.instruments.andor_spectrograph import AndorSpectrograph
        from andor_pymeasure.instruments.delay_stage import MockDelayStage, NewportDelayStage

        # Initialize camera
        self.camera = AndorCamera()
        self.camera.initialize()

        # Initialize spectrograph
        self.spectrograph = AndorSpectrograph()
        self.spectrograph.initialize()

        # Initialize delay stage
        if self.use_mock_stage:
            log.info("Using mock delay stage")
            self.delay_stage = MockDelayStage(
                position_min=0.0,
                position_max=300.0,
                velocity=50.0,  # Fast for testing
            )
        else:
            # Use real Newport stage - configure port as needed
            self.delay_stage = NewportDelayStage(port="COM1")

        self.delay_stage.initialize()

        log.info(f"Camera detector: {self.camera.xpixels}x{self.camera.ypixels}")
        log.info(f"Delay stage range: {self.delay_stage.delay_range_ps} ps")

        # Enable cooler if requested
        if self.cooler_enabled:
            log.info(f"Enabling cooler, target: {self.target_temperature}C")
            self.camera.cooler_on(target=self.target_temperature)

    def execute(self):
        """Run pump-probe scan."""
        # Set spectrograph
        log.info(f"Setting grating to {self.grating}")
        self.spectrograph.grating = self.grating

        if self.should_stop():
            return

        log.info(f"Setting center wavelength to {self.center_wavelength}nm")
        self.spectrograph.wavelength = self.center_wavelength

        if self.should_stop():
            return

        # Get wavelength calibration
        wavelengths = self.spectrograph.get_calibration(
            self.camera.xpixels,
            self.camera.info.pixel_width,
        )
        log.info(f"Wavelength range: {wavelengths[0]:.2f} - {wavelengths[-1]:.2f} nm")

        # Generate delay positions
        delays = np.arange(self.delay_start, self.delay_end + self.delay_step, self.delay_step)
        num_positions = len(delays)
        log.info(f"Scanning {num_positions} delay positions from {self.delay_start} to {self.delay_end} ps")

        # Set camera exposure
        self.camera.set_exposure(self.exposure_time)

        # Scan loop
        for i, delay in enumerate(delays):
            if self.should_stop():
                log.warning("Scan aborted by user")
                return

            log.info(f"Delay {i+1}/{num_positions}: {delay:.2f} ps")

            # Move delay stage
            self.delay_stage.position_ps = delay

            if self.should_stop():
                return

            # Acquire spectrum (with accumulations if requested)
            if self.num_accumulations > 1:
                accumulated = np.zeros(self.camera.xpixels, dtype=np.float64)
                for j in range(self.num_accumulations):
                    if self.should_stop():
                        return
                    accumulated += self.camera.acquire_fvb()
                spectrum = accumulated / self.num_accumulations
            else:
                spectrum = self.camera.acquire_fvb()

            # Emit data points
            for wl, intensity in zip(wavelengths, spectrum):
                self.emit(
                    "results",
                    {
                        "Delay": delay,
                        "Wavelength": wl,
                        "Intensity": intensity,
                    },
                )

            # Update progress
            self.emit("progress", 100 * (i + 1) / num_positions)

        log.info("Pump-probe scan complete")

    def shutdown(self):
        """Cleanup hardware."""
        log.info("Shutting down hardware...")

        if hasattr(self, "delay_stage"):
            self.delay_stage.shutdown()

        if hasattr(self, "camera"):
            self.camera.shutdown()

        if hasattr(self, "spectrograph"):
            self.spectrograph.shutdown()

        log.info("Hardware shutdown complete")


class PumpProbeImageProcedure(Procedure):
    """Pump-probe 2D image procedure.

    This procedure scans a delay stage while acquiring 2D images
    at each delay position. The result is a 3D dataset of intensity vs
    delay time, wavelength, and spatial position.
    """

    # Delay parameters
    delay_start = FloatParameter(
        "Delay Start",
        units="ps",
        default=-10.0,
    )
    delay_end = FloatParameter(
        "Delay End",
        units="ps",
        default=100.0,
    )
    delay_step = FloatParameter(
        "Delay Step",
        units="ps",
        default=1.0,
        minimum=0.001,
    )

    # Spectrograph parameters
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

    # Camera parameters
    exposure_time = FloatParameter(
        "Exposure Time",
        units="s",
        default=1.0,
        minimum=0.001,
        maximum=600.0,
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

    # Cooling
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

    # Mock mode for testing
    use_mock_stage = BooleanParameter(
        "Use Mock Stage",
        default=False,
    )

    DATA_COLUMNS = ["Delay", "Wavelength", "Y_Position", "Intensity"]

    def startup(self):
        """Initialize hardware."""
        log.info("Initializing hardware for pump-probe image experiment...")

        from andor_pymeasure.instruments.andor_camera import AndorCamera
        from andor_pymeasure.instruments.andor_spectrograph import AndorSpectrograph
        from andor_pymeasure.instruments.delay_stage import MockDelayStage, NewportDelayStage

        # Initialize camera
        self.camera = AndorCamera()
        self.camera.initialize()

        # Initialize spectrograph
        self.spectrograph = AndorSpectrograph()
        self.spectrograph.initialize()

        # Initialize delay stage
        if self.use_mock_stage:
            log.info("Using mock delay stage")
            self.delay_stage = MockDelayStage(
                position_min=0.0,
                position_max=300.0,
                velocity=50.0,
            )
        else:
            self.delay_stage = NewportDelayStage(port="COM1")

        self.delay_stage.initialize()

        log.info(f"Camera detector: {self.camera.xpixels}x{self.camera.ypixels}")

        # Enable cooler if requested
        if self.cooler_enabled:
            log.info(f"Enabling cooler, target: {self.target_temperature}C")
            self.camera.cooler_on(target=self.target_temperature)

    def execute(self):
        """Run pump-probe image scan."""
        # Set spectrograph
        log.info(f"Setting grating to {self.grating}")
        self.spectrograph.grating = self.grating

        if self.should_stop():
            return

        log.info(f"Setting center wavelength to {self.center_wavelength}nm")
        self.spectrograph.wavelength = self.center_wavelength

        if self.should_stop():
            return

        # Get wavelength calibration (accounting for binning)
        wavelengths = self.spectrograph.get_binned_calibration(
            self.camera.xpixels, self.camera.info.pixel_width, self.hbin
        )

        # Generate delay positions
        delays = np.arange(self.delay_start, self.delay_end + self.delay_step, self.delay_step)
        num_positions = len(delays)
        log.info(f"Scanning {num_positions} delay positions with 2D images")

        # Set camera exposure
        self.camera.set_exposure(self.exposure_time)

        # Scan loop
        for i, delay in enumerate(delays):
            if self.should_stop():
                log.warning("Scan aborted by user")
                return

            log.info(f"Delay {i+1}/{num_positions}: {delay:.2f} ps")

            # Move delay stage
            self.delay_stage.position_ps = delay

            if self.should_stop():
                return

            # Acquire image
            image = self.camera.acquire_image(hbin=self.hbin, vbin=self.vbin)

            # Emit data points
            for y_idx in range(image.shape[0]):
                for x_idx in range(image.shape[1]):
                    self.emit(
                        "results",
                        {
                            "Delay": delay,
                            "Wavelength": wavelengths[x_idx],
                            "Y_Position": y_idx,
                            "Intensity": image[y_idx, x_idx],
                        },
                    )

            # Update progress
            self.emit("progress", 100 * (i + 1) / num_positions)

        log.info("Pump-probe image scan complete")

    def shutdown(self):
        """Cleanup hardware."""
        log.info("Shutting down hardware...")

        if hasattr(self, "delay_stage"):
            self.delay_stage.shutdown()

        if hasattr(self, "camera"):
            self.camera.shutdown()

        if hasattr(self, "spectrograph"):
            self.spectrograph.shutdown()

        log.info("Hardware shutdown complete")
