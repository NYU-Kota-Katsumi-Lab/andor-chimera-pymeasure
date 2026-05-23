"""Configuration classes for the Qt GUI.

This module provides data classes for application configuration,
including hardware settings, UI preferences, and calibration options.
"""

from __future__ import annotations

import logging
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ChopperConfig:
    """Configuration for the Thorlabs MC2000B chopper pair.

    Attributes:
        enabled: Whether to attempt opening the chopper serial ports during
            hardware init. If False, ``HardwareManager.chopper_pair`` is None.
        pump_port: COM port for the pump-master chopper (default ``COM4``).
        probe_port: COM port for the probe-slave chopper (default ``COM3``).
            If empty, only the pump chopper is opened.
        blade: Blade name for both choppers (default ``MC1F10HP``).
        frequency_hz: Target pump frequency in Hz (default 250).
    """

    enabled: bool = True
    pump_port: str = "COM4"
    probe_port: str = "COM3"
    blade: str = "MC1F10HP"
    frequency_hz: int = 250


@dataclass(frozen=True)
class HardwareConfig:
    """Configuration for hardware devices.

    Attributes:
        sdk_path: Path to the Andor SDK installation.
        mock_mode: Whether to use mock hardware for development.
        default_temperature: Default cooling target temperature (°C).
        warmup_temperature: Temperature to warm up to before shutdown (°C).
    """

    sdk_path: str = r"C:\Program Files\Andor SDK"
    mock_mode: bool = False
    default_temperature: int = -60
    warmup_temperature: int = -20


@dataclass(frozen=True)
class UIConfig:
    """Configuration for the user interface.

    Attributes:
        window_title: Main window title.
        temperature_poll_interval_ms: Temperature polling interval in ms.
    """

    window_title: str = "Andor Spectrometer Control"
    temperature_poll_interval_ms: int = 2000


@dataclass(frozen=True)
class CalibrationConfig:
    """Configuration for wavelength calibration.

    Attributes:
        source: Calibration source - "sdk" for SDK-computed or "file" for file.
        file_path: Path to calibration file (when source is "file").
    """

    source: str = "sdk"
    file_path: Optional[str] = None


@dataclass
class AppConfig:
    """Main application configuration.

    This is the top-level configuration container that holds all
    configuration sections. It is mutable to allow runtime updates.

    Attributes:
        hardware: Hardware device configuration.
        ui: User interface configuration.
        calibration: Calibration settings.
        motion_controllers: Motion controller configuration dict passed
            directly to MotionControllerManager. When absent or None,
            HardwareManager falls back to a single mock delay stage.
    """

    hardware: HardwareConfig = field(default_factory=HardwareConfig)
    ui: UIConfig = field(default_factory=UIConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    motion_controllers: Optional[dict] = field(default=None)
    choppers: ChopperConfig = field(default_factory=ChopperConfig)

    @classmethod
    def default(cls) -> "AppConfig":
        """Create a configuration with default values.

        Returns:
            AppConfig with all default settings.
        """
        return cls(
            hardware=HardwareConfig(),
            ui=UIConfig(),
            calibration=CalibrationConfig(),
        )

    def to_yaml(self, path: Path) -> None:
        """Save configuration to a YAML file.

        Args:
            path: Path to the YAML file to write.
        """
        data = {
            "hardware": asdict(self.hardware),
            "ui": asdict(self.ui),
            "calibration": asdict(self.calibration),
            "choppers": asdict(self.choppers),
        }
        if self.motion_controllers is not None:
            data["motion_controllers"] = self.motion_controllers

        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False)

    @classmethod
    def from_yaml(cls, path: Path) -> "AppConfig":
        """Load configuration from a YAML file.

        Args:
            path: Path to the YAML file to read.

        Returns:
            AppConfig loaded from the file.
        """
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        return cls._from_dict(data)

    @classmethod
    def _from_dict(cls, data: dict[str, Any]) -> "AppConfig":
        """Create AppConfig from a dictionary.

        Args:
            data: Dictionary with config data.

        Returns:
            AppConfig instance.
        """
        hardware_data = data.get("hardware", {})
        ui_data = data.get("ui", {})
        calibration_data = data.get("calibration", {})
        chopper_data = data.get("choppers", {})

        return cls(
            hardware=HardwareConfig(**hardware_data) if hardware_data else HardwareConfig(),
            ui=UIConfig(**ui_data) if ui_data else UIConfig(),
            calibration=CalibrationConfig(**calibration_data) if calibration_data else CalibrationConfig(),
            motion_controllers=data.get("motion_controllers"),
            choppers=ChopperConfig(**chopper_data) if chopper_data else ChopperConfig(),
        )

    @classmethod
    def load_or_default(cls, path: Optional[Path] = None) -> "AppConfig":
        """Load configuration from file or return defaults.

        Also applies environment variable overrides.

        Args:
            path: Path to config file (optional).

        Returns:
            AppConfig loaded from file or with defaults.
        """
        if path is not None and path.exists():
            try:
                config = cls.from_yaml(path)
            except Exception as e:
                log.warning(f"Failed to load config from {path}: {e}")
                config = cls.default()
        else:
            config = cls.default()

        # Apply environment variable overrides
        config = cls._apply_env_overrides(config)

        return config

    @classmethod
    def _apply_env_overrides(cls, config: "AppConfig") -> "AppConfig":
        """Apply environment variable overrides to config.

        Args:
            config: Original configuration.

        Returns:
            Configuration with env overrides applied.
        """
        mock_mode = os.environ.get("ANDOR_MOCK", "0") == "1"

        if mock_mode != config.hardware.mock_mode:
            # Need to recreate HardwareConfig since it's frozen
            config.hardware = HardwareConfig(
                sdk_path=config.hardware.sdk_path,
                mock_mode=mock_mode,
                default_temperature=config.hardware.default_temperature,
                warmup_temperature=config.hardware.warmup_temperature,
            )

        return config
