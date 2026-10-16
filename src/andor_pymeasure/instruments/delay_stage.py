"""Delay stage instrument for pump-probe experiments.

This module provides a base class and implementations for delay stages
used in pump-probe spectroscopy experiments.

The delay stage position can be specified in either:
- Physical units (mm) - actual motor position
- Time units (ps) - optical delay time

Conversion: delay_ps = (2 * position_mm) / (c * 1e-12)
where c = 299.792458 mm/ns (speed of light)
"""

from __future__ import annotations

import logging
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())

# Speed of light in mm/ps
SPEED_OF_LIGHT_MM_PS = 0.299792458  # mm/ps


@dataclass(frozen=True)
class DelayStageInfo:
    """Delay stage information."""

    model: str = ""
    serial_number: str = ""
    position_min: float = 0.0  # mm
    position_max: float = 0.0  # mm
    velocity_max: float = 0.0  # mm/s


class DelayStage(ABC):
    """Abstract base class for delay stages.

    This class provides the interface for controlling delay stages in
    pump-probe experiments. Subclasses should implement the abstract
    methods for specific hardware.

    The stage position can be accessed/set in either mm or ps units.
    """

    def __init__(self):
        self._initialized = False
        self._info: Optional[DelayStageInfo] = None
        self._lock = threading.Lock()

    @property
    def info(self) -> Optional[DelayStageInfo]:
        """Get delay stage information."""
        return self._info

    @property
    @abstractmethod
    def position_mm(self) -> float:
        """Get current position in mm."""
        pass

    @position_mm.setter
    @abstractmethod
    def position_mm(self, value: float) -> None:
        """Set position in mm."""
        pass

    @property
    def position_ps(self) -> float:
        """Get current position in ps (optical delay)."""
        # Round-trip delay: light travels to mirror and back
        return (2 * self.position_mm) / SPEED_OF_LIGHT_MM_PS

    @position_ps.setter
    def position_ps(self, value: float) -> None:
        """Set position in ps (optical delay)."""
        self.position_mm = (value * SPEED_OF_LIGHT_MM_PS) / 2

    @property
    def delay_range_ps(self) -> tuple[float, float]:
        """Get delay range in ps."""
        if self._info:
            min_ps = (2 * self._info.position_min) / SPEED_OF_LIGHT_MM_PS
            max_ps = (2 * self._info.position_max) / SPEED_OF_LIGHT_MM_PS
            return (min_ps, max_ps)
        return (0.0, 0.0)

    @abstractmethod
    def initialize(self) -> None:
        """Initialize the delay stage."""
        pass

    @abstractmethod
    def home(self) -> None:
        """Home the delay stage."""
        pass

    @abstractmethod
    def stop(self) -> None:
        """Stop any motion."""
        pass

    @abstractmethod
    def is_moving(self) -> bool:
        """Check if stage is moving."""
        pass

    @abstractmethod
    def shutdown(self) -> None:
        """Shutdown the delay stage."""
        pass


class MockDelayStage(DelayStage):
    """Mock delay stage for testing without hardware.

    This implementation simulates a delay stage with configurable
    position limits and movement time.
    """

    _MOVE_POLL_S = 0.005  # how often a simulated move checks for stop()

    def __init__(
        self,
        position_min: float = 0.0,
        position_max: float = 300.0,
        velocity: float = 10.0,
    ):
        """Initialize mock delay stage.

        Args:
            position_min: Minimum position in mm.
            position_max: Maximum position in mm.
            velocity: Movement velocity in mm/s.
        """
        super().__init__()
        self._position_min = position_min
        self._position_max = position_max
        self._velocity = velocity
        self._position = 0.0
        self._moving = False
        self._target = 0.0
        self._move_thread: Optional[threading.Thread] = None

    @property
    def position_mm(self) -> float:
        """Get current position in mm."""
        return self._position

    @position_mm.setter
    def position_mm(self, value: float) -> None:
        """Set position in mm (blocking)."""
        if not self._initialized:
            raise RuntimeError("Delay stage not initialized")

        # Clamp to limits
        value = max(self._position_min, min(value, self._position_max))

        with self._lock:
            self._target = value
            self._moving = True
            start = self._position

        # Simulate movement time in short steps so stop() can interrupt the move
        distance = abs(value - start)
        move_time = distance / self._velocity

        log.debug(f"Moving to {value}mm (distance: {distance:.2f}mm, time: {move_time:.2f}s)")
        t0 = time.monotonic()
        while True:
            elapsed = time.monotonic() - t0
            with self._lock:
                if not self._moving:  # stop() called: halt where the stage is now
                    frac = min(elapsed / move_time, 1.0) if move_time > 0 else 1.0
                    self._position = start + (value - start) * frac
                    log.debug(f"Move stopped: position = {self._position}mm")
                    return
                if elapsed >= move_time:
                    self._position = value
                    self._moving = False
                    break
            time.sleep(min(self._MOVE_POLL_S, move_time - elapsed))

        log.debug(f"Move complete: position = {self._position}mm")

    def initialize(self) -> None:
        """Initialize the delay stage."""
        if self._initialized:
            log.warning("Delay stage already initialized")
            return

        self._info = DelayStageInfo(
            model="Mock Delay Stage",
            serial_number="MOCK-001",
            position_min=self._position_min,
            position_max=self._position_max,
            velocity_max=self._velocity,
        )

        self._initialized = True
        log.info(
            f"Mock delay stage initialized: "
            f"range {self._position_min} - {self._position_max} mm"
        )

    def home(self) -> None:
        """Home the delay stage (move to position 0)."""
        if not self._initialized:
            raise RuntimeError("Delay stage not initialized")

        log.info("Homing delay stage...")
        self.position_mm = 0.0
        log.info("Home complete")

    def stop(self) -> None:
        """Stop any motion."""
        with self._lock:
            self._moving = False
            self._target = self._position

    def is_moving(self) -> bool:
        """Check if stage is moving."""
        return self._moving

    def shutdown(self) -> None:
        """Shutdown the delay stage."""
        if not self._initialized:
            return

        self.stop()
        self._initialized = False
        log.info("Delay stage shutdown complete")


class NewportDelayStage(DelayStage):
    """Newport SMC100 delay stage controller.

    This implementation interfaces with Newport SMC100 series motion
    controllers via serial port.

    Note: Requires pyserial package.
    """

    def __init__(
        self,
        port: str = "COM1",
        controller_address: int = 1,
        baudrate: int = 57600,
    ):
        """Initialize Newport delay stage.

        Args:
            port: Serial port (e.g., "COM1" or "/dev/ttyUSB0").
            controller_address: Controller address (default 1).
            baudrate: Serial baud rate (default 57600).
        """
        super().__init__()
        self._port = port
        self._address = controller_address
        self._baudrate = baudrate
        self._serial = None

    def _send_command(self, command: str) -> str:
        """Send command and get response."""
        if self._serial is None:
            raise RuntimeError("Serial port not open")

        # Format: {address}{command}\r\n
        full_cmd = f"{self._address}{command}\r\n"
        self._serial.write(full_cmd.encode())

        # Read response
        response = self._serial.readline().decode().strip()
        return response

    @property
    def position_mm(self) -> float:
        """Get current position in mm."""
        if not self._initialized:
            return 0.0

        with self._lock:
            response = self._send_command("TP")
            # Response format: {address}TP{position}
            try:
                pos_str = response.split("TP")[1]
                return float(pos_str)
            except (IndexError, ValueError):
                log.error(f"Failed to parse position: {response}")
                return 0.0

    @position_mm.setter
    def position_mm(self, value: float) -> None:
        """Set position in mm (blocking)."""
        if not self._initialized:
            raise RuntimeError("Delay stage not initialized")

        with self._lock:
            # Send move command
            self._send_command(f"PA{value}")

        # Wait for motion to complete
        while self.is_moving():
            time.sleep(0.1)

    def initialize(self) -> None:
        """Initialize the delay stage."""
        if self._initialized:
            log.warning("Delay stage already initialized")
            return

        import serial

        self._serial = serial.Serial(
            port=self._port,
            baudrate=self._baudrate,
            timeout=1.0,
        )

        # Get stage info
        with self._lock:
            # Get model
            model = self._send_command("ID?")

            # Get travel limits
            response = self._send_command("SL?")
            pos_min = float(response.split("SL")[1]) if "SL" in response else 0.0

            response = self._send_command("SR?")
            pos_max = float(response.split("SR")[1]) if "SR" in response else 300.0

            response = self._send_command("VA?")
            velocity = float(response.split("VA")[1]) if "VA" in response else 10.0

        self._info = DelayStageInfo(
            model=model,
            serial_number=f"Newport-{self._address}",
            position_min=pos_min,
            position_max=pos_max,
            velocity_max=velocity,
        )

        self._initialized = True
        log.info(f"Newport delay stage initialized: {model}")

    def home(self) -> None:
        """Home the delay stage."""
        if not self._initialized:
            raise RuntimeError("Delay stage not initialized")

        log.info("Homing delay stage...")
        with self._lock:
            self._send_command("OR")

        # Wait for home to complete
        while self.is_moving():
            time.sleep(0.1)

        log.info("Home complete")

    def stop(self) -> None:
        """Stop any motion."""
        if self._serial is not None:
            with self._lock:
                self._send_command("ST")

    def is_moving(self) -> bool:
        """Check if stage is moving."""
        if not self._initialized:
            return False

        with self._lock:
            response = self._send_command("TS")
            # Check motion status bit
            try:
                status = response.split("TS")[1]
                # Status code ending in 28 or 29 indicates moving
                return status.endswith("28") or status.endswith("29")
            except (IndexError, ValueError):
                return False

    def shutdown(self) -> None:
        """Shutdown the delay stage."""
        if not self._initialized:
            return

        self.stop()

        if self._serial is not None:
            self._serial.close()
            self._serial = None

        self._initialized = False
        log.info("Delay stage shutdown complete")
