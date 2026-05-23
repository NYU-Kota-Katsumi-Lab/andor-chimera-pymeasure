"""Thorlabs MC2000B optical chopper driver (raw serial).

Uses the ASCII command set documented in the MC2000B manual chapter 8 over a
115200 8N1 serial connection. **Does not** use the Thorlabs DLL — that DLL
times out (-3) against firmware v4.19. The raw protocol works reliably.

A typical pump/probe pair is configured as:

    pump  = MC2000B("COM4")    # master, INT-INNER reference
    probe = MC2000B("COM3")    # slave,  EXT-INNER reference, harmonic N/D=1/1

Use ``configure_pump_master()`` and ``configure_probe_slave()`` to apply the
canonical pair configuration; or set parameters individually via the
properties.

``MockMC2000B`` is a drop-in replacement for offline tests and ``ANDOR_MOCK=1``
mode. It tracks state but does no serial IO.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Optional

log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())

#: Default baud rate (manual chapter 8: 115200 8N1)
BAUD_RATE = 115200

#: Default per-command timeout (replies typically arrive in tens of ms)
DEFAULT_TIMEOUT_S = 0.5

#: Blade index → name lookup (manual section 8.2)
BLADE_INDEX = {
    "MC1F2": 0,
    "MC1F10": 1,
    "MC1F15": 2,
    "MC1F30": 3,
    "MC1F60": 4,
    "MC1F100": 5,
    "MC1F10HP": 6,
    "MC1F2P10": 7,
    "MC1F6P10": 8,
    "MC1F10A": 9,
    "MC2F330": 10,
    "MC2F47": 11,
    "MC2F57B": 12,
    "MC2F860": 13,
    "MC2F5360": 14,
}

#: Ref-input modes for high-precision blades (MC1F10HP, MC1F2, MC1F2P10).
#: For simpler blades, only 0=internal / 1=external are valid.
REF_INPUT_HIGHPREC = {
    "INT-OUTER": 0,
    "INT-INNER": 1,
    "EXT-OUTER": 2,
    "EXT-INNER": 3,
}


# ---------------------------------------------------------------------------
# Hardware driver
# ---------------------------------------------------------------------------


class MC2000B:
    """Single MC2000B chopper over a serial port.

    Args:
        port: COM port name (e.g. ``"COM3"``).
        timeout: Per-command serial timeout in seconds.
        open_on_init: If True (default), open the serial port immediately.
            If False, call ``open()`` manually.

    Example::

        with MC2000B("COM4") as ch:
            ch.blade = "MC1F10HP"
            ch.reference = "INT-INNER"
            ch.frequency_hz = 250
            ch.enabled = True
    """

    def __init__(
        self,
        port: str,
        timeout: float = DEFAULT_TIMEOUT_S,
        open_on_init: bool = True,
    ):
        self._port = port
        self._timeout = timeout
        self._ser = None  # type: ignore[var-annotated]
        self._lock = threading.Lock()
        if open_on_init:
            self.open()

    # -- lifecycle ---------------------------------------------------------

    def open(self) -> None:
        """Open the serial port."""
        import serial  # imported lazily so tests can stub it out

        if self._ser is not None and getattr(self._ser, "is_open", False):
            return
        self._ser = serial.Serial(
            self._port,
            BAUD_RATE,
            bytesize=8,
            parity="N",
            stopbits=1,
            timeout=self._timeout,
            write_timeout=self._timeout,
        )
        # Drain any leftover banner/prompt
        time.sleep(0.05)
        try:
            self._ser.reset_input_buffer()
        except Exception:
            pass

    def close(self) -> None:
        """Close the serial port."""
        ser = self._ser
        self._ser = None
        if ser is not None:
            try:
                ser.close()
            except Exception:
                pass

    def __enter__(self) -> "MC2000B":
        self.open()
        return self

    def __exit__(self, *_a) -> None:
        self.close()

    @property
    def port(self) -> str:
        """Return the COM port name."""
        return self._port

    @property
    def is_open(self) -> bool:
        """Return True if the serial port is open."""
        return self._ser is not None and getattr(self._ser, "is_open", False)

    # -- low-level IO ------------------------------------------------------

    def _query(self, cmd: str) -> str:
        """Send a command, return the response body (echo + prompt stripped).

        Raises:
            RuntimeError: If the port is closed or the device returned an error.
        """
        if self._ser is None or not self.is_open:
            raise RuntimeError(f"MC2000B({self._port!r}) port is not open")

        with self._lock:
            self._ser.reset_input_buffer()
            self._ser.write((cmd + "\r").encode("ascii"))
            buf = b""
            deadline = time.time() + self._timeout * 4
            while time.time() < deadline:
                chunk = self._ser.read(256)
                if chunk:
                    buf += chunk
                    if buf.endswith(b"> "):
                        break
                else:
                    time.sleep(0.01)
            text = buf.decode("ascii", errors="replace")

        lines = [ln for ln in text.replace("\r", "\n").split("\n") if ln.strip()]
        if lines and lines[0].strip() == cmd:
            lines = lines[1:]
        if lines and lines[-1].strip() == ">":
            lines = lines[:-1]
        result = "\n".join(lines).strip()
        if "command error" in result.lower() or "cmd_arg_invalid" in result.lower():
            raise RuntimeError(f"MC2000B({self._port!r}) rejected {cmd!r}: {result}")
        return result

    def query(self, cmd: str) -> str:
        """Send a raw query command and return the response text."""
        return self._query(cmd)

    def _set(self, key: str, value: int) -> None:
        self._query(f"{key}={int(value)}")

    def _get_int(self, key: str) -> int:
        resp = self._query(f"{key}?")
        try:
            return int(resp.strip())
        except ValueError as e:
            raise RuntimeError(f"{key}? returned non-int: {resp!r}") from e

    # -- high-level properties --------------------------------------------

    @property
    def id(self) -> str:
        """Return the device ID string (model + firmware version)."""
        return self._query("id?")

    @property
    def enabled(self) -> bool:
        """Whether the chopper motor is enabled (running)."""
        return bool(self._get_int("enable"))

    @enabled.setter
    def enabled(self, value: bool) -> None:
        self._set("enable", 1 if value else 0)

    @property
    def blade(self) -> int:
        """Blade type index (see :data:`BLADE_INDEX`)."""
        return self._get_int("blade")

    @blade.setter
    def blade(self, value) -> None:
        if isinstance(value, str):
            idx = BLADE_INDEX.get(value.upper())
            if idx is None:
                raise ValueError(
                    f"Unknown blade {value!r}. Valid: {sorted(BLADE_INDEX)}"
                )
            value = idx
        self._set("blade", value)

    @property
    def reference(self) -> int:
        """Reference input mode (see :data:`REF_INPUT_HIGHPREC`)."""
        return self._get_int("ref")

    @reference.setter
    def reference(self, value) -> None:
        if isinstance(value, str):
            idx = REF_INPUT_HIGHPREC.get(value.upper())
            if idx is None:
                raise ValueError(
                    f"Unknown reference mode {value!r}. "
                    f"Valid: {sorted(REF_INPUT_HIGHPREC)}"
                )
            value = idx
        self._set("ref", value)

    @property
    def reference_output(self) -> int:
        """Reference output selector. 1=outer, 2=inner (see manual 8.4)."""
        return self._get_int("output")

    @reference_output.setter
    def reference_output(self, value: int) -> None:
        self._set("output", value)

    @property
    def n_harmonic(self) -> int:
        """Harmonic multiplier N (frequency ratio numerator)."""
        return self._get_int("nharmonic")

    @n_harmonic.setter
    def n_harmonic(self, value: int) -> None:
        self._set("nharmonic", value)

    @property
    def d_harmonic(self) -> int:
        """Harmonic divider D (frequency ratio denominator)."""
        return self._get_int("dharmonic")

    @d_harmonic.setter
    def d_harmonic(self, value: int) -> None:
        self._set("dharmonic", value)

    @property
    def phase_deg(self) -> int:
        """Phase offset in degrees. Firmware v4.19 only accepts integer 0-360."""
        return self._get_int("phase")

    @phase_deg.setter
    def phase_deg(self, value) -> None:
        deg = int(round(float(value))) % 360
        self._set("phase", deg)

    @property
    def frequency_hz(self) -> int:
        """Target frequency in Hz (when using INT reference)."""
        return self._get_int("freq")

    @frequency_hz.setter
    def frequency_hz(self, value: int) -> None:
        self._set("freq", int(value))

    @property
    def actual_frequency_hz(self) -> float:
        """Actual blade frequency in Hz (from refoutfreq)."""
        resp = self._query("refoutfreq?")
        try:
            return float(resp.strip())
        except ValueError as e:
            raise RuntimeError(f"refoutfreq? returned non-float: {resp!r}") from e

    @property
    def input_frequency_hz(self) -> float:
        """External reference input frequency in Hz (from input?)."""
        resp = self._query("input?")
        try:
            return float(resp.strip())
        except ValueError as e:
            raise RuntimeError(f"input? returned non-float: {resp!r}") from e

    # -- snapshot ----------------------------------------------------------

    def snapshot(self) -> dict:
        """Return a dict of all readable parameters (best-effort).

        Fields that fail to read are returned as the error string instead.
        """
        out: dict = {"port": self._port}
        readers = {
            "id": lambda: self.id,
            "enabled": lambda: self.enabled,
            "blade": lambda: self.blade,
            "reference": lambda: self.reference,
            "reference_output": lambda: self.reference_output,
            "n_harmonic": lambda: self.n_harmonic,
            "d_harmonic": lambda: self.d_harmonic,
            "phase_deg": lambda: self.phase_deg,
            "frequency_hz": lambda: self.frequency_hz,
            "actual_frequency_hz": lambda: self.actual_frequency_hz,
            "input_frequency_hz": lambda: self.input_frequency_hz,
        }
        for name, fn in readers.items():
            try:
                out[name] = fn()
            except Exception as e:
                out[name] = f"ERROR ({e})"
        return out

    # -- pair configuration helpers ----------------------------------------

    def configure_pump_master(
        self,
        blade: str = "MC1F10HP",
        frequency_hz: int = 250,
        enable: bool = True,
    ) -> None:
        """Configure this chopper as the pump master (internal reference).

        Args:
            blade: Blade type name (default ``MC1F10HP``).
            frequency_hz: Target chopping frequency in Hz.
            enable: If True, enable the chopper after configuring.
        """
        self.blade = blade
        self.reference = "INT-INNER"
        self.reference_output = 1  # outer ref-out so the slave can lock
        self.frequency_hz = frequency_hz
        if enable:
            self.enabled = True
        log.info(
            "Pump master configured on %s: blade=%s ref=INT-INNER freq=%d Hz",
            self._port, blade, frequency_hz,
        )

    def configure_probe_slave(
        self,
        blade: str = "MC1F10HP",
        n_harmonic: int = 1,
        d_harmonic: int = 1,
        phase_deg: int = 0,
        enable: bool = True,
    ) -> None:
        """Configure this chopper as a probe slave (external reference).

        Args:
            blade: Blade type name (default ``MC1F10HP``).
            n_harmonic: Harmonic multiplier N.
            d_harmonic: Harmonic divider D.
            phase_deg: Initial phase offset in degrees.
            enable: If True, enable the chopper after configuring.
        """
        self.blade = blade
        self.reference = "EXT-INNER"
        self.n_harmonic = n_harmonic
        self.d_harmonic = d_harmonic
        self.phase_deg = phase_deg
        if enable:
            self.enabled = True
        log.info(
            "Probe slave configured on %s: blade=%s ref=EXT-INNER N/D=%d/%d phase=%d°",
            self._port, blade, n_harmonic, d_harmonic, phase_deg,
        )


# ---------------------------------------------------------------------------
# Mock
# ---------------------------------------------------------------------------


class MockMC2000B:
    """Mock MC2000B for offline tests. Tracks state in memory."""

    def __init__(self, port: str = "MOCK", **_kwargs):
        self._port = port
        self._state = {
            "enabled": False,
            "blade": BLADE_INDEX["MC1F10HP"],
            "reference": REF_INPUT_HIGHPREC["INT-INNER"],
            "reference_output": 1,
            "n_harmonic": 1,
            "d_harmonic": 1,
            "phase_deg": 0,
            "frequency_hz": 250,
        }

    # lifecycle ------------------------------------------------------------
    def open(self) -> None:
        pass

    def close(self) -> None:
        pass

    def __enter__(self) -> "MockMC2000B":
        return self

    def __exit__(self, *_a) -> None:
        pass

    @property
    def port(self) -> str:
        return self._port

    @property
    def is_open(self) -> bool:
        return True

    # parameters -----------------------------------------------------------
    @property
    def id(self) -> str:
        return "MOCK MC2000B v0.0"

    @property
    def enabled(self) -> bool:
        return self._state["enabled"]

    @enabled.setter
    def enabled(self, value: bool) -> None:
        self._state["enabled"] = bool(value)

    @property
    def blade(self) -> int:
        return self._state["blade"]

    @blade.setter
    def blade(self, value) -> None:
        if isinstance(value, str):
            idx = BLADE_INDEX.get(value.upper())
            if idx is None:
                raise ValueError(f"Unknown blade {value!r}")
            value = idx
        self._state["blade"] = int(value)

    @property
    def reference(self) -> int:
        return self._state["reference"]

    @reference.setter
    def reference(self, value) -> None:
        if isinstance(value, str):
            idx = REF_INPUT_HIGHPREC.get(value.upper())
            if idx is None:
                raise ValueError(f"Unknown reference {value!r}")
            value = idx
        self._state["reference"] = int(value)

    @property
    def reference_output(self) -> int:
        return self._state["reference_output"]

    @reference_output.setter
    def reference_output(self, value: int) -> None:
        self._state["reference_output"] = int(value)

    @property
    def n_harmonic(self) -> int:
        return self._state["n_harmonic"]

    @n_harmonic.setter
    def n_harmonic(self, value: int) -> None:
        self._state["n_harmonic"] = int(value)

    @property
    def d_harmonic(self) -> int:
        return self._state["d_harmonic"]

    @d_harmonic.setter
    def d_harmonic(self, value: int) -> None:
        self._state["d_harmonic"] = int(value)

    @property
    def phase_deg(self) -> int:
        return self._state["phase_deg"]

    @phase_deg.setter
    def phase_deg(self, value) -> None:
        self._state["phase_deg"] = int(round(float(value))) % 360

    @property
    def frequency_hz(self) -> int:
        return self._state["frequency_hz"]

    @frequency_hz.setter
    def frequency_hz(self, value: int) -> None:
        self._state["frequency_hz"] = int(value)

    @property
    def actual_frequency_hz(self) -> float:
        """Mock: report nominal frequency divided by D/N."""
        n = max(1, self._state["n_harmonic"])
        d = max(1, self._state["d_harmonic"])
        return float(self._state["frequency_hz"]) * n / d

    @property
    def input_frequency_hz(self) -> float:
        return 500.0  # nominal external reference

    def query(self, _cmd: str) -> str:
        return ""

    def snapshot(self) -> dict:
        return {
            "port": self._port,
            "id": self.id,
            "enabled": self.enabled,
            "blade": self.blade,
            "reference": self.reference,
            "reference_output": self.reference_output,
            "n_harmonic": self.n_harmonic,
            "d_harmonic": self.d_harmonic,
            "phase_deg": self.phase_deg,
            "frequency_hz": self.frequency_hz,
            "actual_frequency_hz": self.actual_frequency_hz,
            "input_frequency_hz": self.input_frequency_hz,
        }

    # pair helpers ---------------------------------------------------------
    configure_pump_master = MC2000B.configure_pump_master
    configure_probe_slave = MC2000B.configure_probe_slave


# ---------------------------------------------------------------------------
# Pair
# ---------------------------------------------------------------------------


class ChopperPair:
    """A pump-master / probe-slave MC2000B pair.

    Args:
        pump: Pump master chopper (or :class:`MockMC2000B`).
        probe: Probe slave chopper (or :class:`MockMC2000B`).

    Use :meth:`setup_pair` to apply the canonical configuration after
    construction. Either or both ports may be ``None`` if not connected.
    """

    def __init__(
        self,
        pump: Optional["MC2000B"] = None,
        probe: Optional["MC2000B"] = None,
    ):
        self._pump = pump
        self._probe = probe

    @property
    def pump(self) -> Optional["MC2000B"]:
        return self._pump

    @property
    def probe(self) -> Optional["MC2000B"]:
        return self._probe

    def setup_pair(
        self,
        blade: str = "MC1F10HP",
        frequency_hz: int = 250,
        n_harmonic: int = 1,
        d_harmonic: int = 1,
        phase_deg: int = 0,
    ) -> None:
        """Apply the canonical pump-master / probe-slave configuration."""
        if self._pump is not None:
            self._pump.configure_pump_master(blade=blade, frequency_hz=frequency_hz)
        # Let the master settle before the slave tries to lock.
        time.sleep(0.5)
        if self._probe is not None:
            self._probe.configure_probe_slave(
                blade=blade,
                n_harmonic=n_harmonic,
                d_harmonic=d_harmonic,
                phase_deg=phase_deg,
            )

    def shutdown(self) -> None:
        """Disable both choppers and close serial ports."""
        for ch in (self._pump, self._probe):
            if ch is None:
                continue
            try:
                ch.enabled = False
            except Exception as e:
                log.warning("Disable failed on %s: %s", getattr(ch, "port", "?"), e)
            try:
                ch.close()
            except Exception:
                pass
