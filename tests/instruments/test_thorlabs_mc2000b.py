"""Tests for Thorlabs MC2000B chopper driver.

Covers:
- ``MockMC2000B`` round-trip through every property
- ``MC2000B`` over a fake serial.Serial (write/read interaction)
- Phase normalisation (firmware accepts integer 0-360 only)
- ``configure_pump_master`` / ``configure_probe_slave`` command sequences
- ``ChopperPair.setup_pair`` and ``shutdown``
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from andor_pymeasure.instruments.thorlabs_mc2000b import (
    BLADE_INDEX,
    REF_INPUT_HIGHPREC,
    ChopperPair,
    MC2000B,
    MockMC2000B,
)


# ---------------------------------------------------------------------------
# MockMC2000B
# ---------------------------------------------------------------------------


class TestMockMC2000B:
    def test_defaults_match_pump_master_config(self):
        ch = MockMC2000B()
        assert ch.blade == BLADE_INDEX["MC1F10HP"]
        assert ch.reference == REF_INPUT_HIGHPREC["INT-INNER"]
        assert ch.frequency_hz == 250
        assert ch.n_harmonic == 1
        assert ch.d_harmonic == 1
        assert ch.enabled is False

    def test_enable_round_trip(self):
        ch = MockMC2000B()
        ch.enabled = True
        assert ch.enabled is True

    def test_phase_normalises_to_integer_modulo_360(self):
        ch = MockMC2000B()
        ch.phase_deg = 19.5  # fractional input
        assert ch.phase_deg == 20  # round to nearest int
        ch.phase_deg = 365
        assert ch.phase_deg == 5  # mod 360
        ch.phase_deg = -1
        assert ch.phase_deg == 359  # python mod is always non-negative

    def test_blade_by_name(self):
        ch = MockMC2000B()
        ch.blade = "MC1F10"
        assert ch.blade == BLADE_INDEX["MC1F10"]

    def test_blade_unknown_name_raises(self):
        ch = MockMC2000B()
        with pytest.raises(ValueError, match="Unknown blade"):
            ch.blade = "NOT_A_BLADE"

    def test_reference_by_name(self):
        ch = MockMC2000B()
        ch.reference = "EXT-INNER"
        assert ch.reference == REF_INPUT_HIGHPREC["EXT-INNER"]

    def test_reference_unknown_name_raises(self):
        ch = MockMC2000B()
        with pytest.raises(ValueError, match="Unknown reference"):
            ch.reference = "BAD"

    def test_actual_frequency_tracks_harmonic(self):
        ch = MockMC2000B()
        ch.frequency_hz = 500
        ch.n_harmonic = 1
        ch.d_harmonic = 2
        assert ch.actual_frequency_hz == pytest.approx(250.0)

    def test_snapshot_returns_all_fields(self):
        ch = MockMC2000B(port="COM_TEST")
        snap = ch.snapshot()
        assert snap["port"] == "COM_TEST"
        assert "enabled" in snap
        assert "phase_deg" in snap
        assert "actual_frequency_hz" in snap
        assert "id" in snap

    def test_configure_pump_master_sets_all_params(self):
        ch = MockMC2000B()
        ch.configure_pump_master(blade="MC1F10HP", frequency_hz=250)
        assert ch.blade == BLADE_INDEX["MC1F10HP"]
        assert ch.reference == REF_INPUT_HIGHPREC["INT-INNER"]
        assert ch.frequency_hz == 250
        assert ch.enabled is True

    def test_configure_probe_slave_sets_all_params(self):
        ch = MockMC2000B()
        ch.configure_probe_slave(
            blade="MC1F10HP", n_harmonic=1, d_harmonic=2, phase_deg=20
        )
        assert ch.reference == REF_INPUT_HIGHPREC["EXT-INNER"]
        assert ch.n_harmonic == 1
        assert ch.d_harmonic == 2
        assert ch.phase_deg == 20
        assert ch.enabled is True


# ---------------------------------------------------------------------------
# MC2000B (real driver) over a fake serial.Serial
# ---------------------------------------------------------------------------


class FakeSerial:
    """Minimal stand-in for serial.Serial.

    Each command in ``responses`` (a dict ``{command: reply_body}``) gets the
    canonical echo + reply + ``"\\r> "`` prompt framing.
    """

    def __init__(self, responses=None):
        self.responses = responses or {}
        self.writes: list[bytes] = []
        self._read_buf = b""
        self.is_open = True

    def reset_input_buffer(self):
        self._read_buf = b""

    def write(self, data: bytes) -> int:
        self.writes.append(data)
        cmd = data.decode("ascii").strip()
        body = self.responses.get(cmd, "")
        # Match the device framing: echo, optional reply body, prompt "> "
        framed = cmd + "\r"
        if body:
            framed += body + "\r"
        framed += "> "
        self._read_buf += framed.encode("ascii")
        return len(data)

    def read(self, n: int = 1) -> bytes:
        if not self._read_buf:
            return b""
        out = self._read_buf[:n]
        self._read_buf = self._read_buf[n:]
        return out

    def close(self):
        self.is_open = False


def _make_chopper(responses):
    """Construct an MC2000B wired to a FakeSerial returning canned responses."""
    fake = FakeSerial(responses)
    import sys
    import types

    if "serial" not in sys.modules:
        stub = types.ModuleType("serial")
        stub.Serial = MagicMock(return_value=fake)
        sys.modules["serial"] = stub
    else:
        sys.modules["serial"].Serial = MagicMock(return_value=fake)
    ch = MC2000B("COM_FAKE")
    return ch, fake


class TestMC2000BSerial:
    def test_phase_setter_sends_integer(self):
        ch, fake = _make_chopper({"phase=20": ""})
        ch.phase_deg = 19.6  # should round to 20
        assert b"phase=20\r" in fake.writes

    def test_phase_getter_parses_int(self):
        ch, _fake = _make_chopper({"phase?": "20"})
        assert ch.phase_deg == 20

    def test_enable_setter(self):
        ch, fake = _make_chopper({"enable=1": ""})
        ch.enabled = True
        assert b"enable=1\r" in fake.writes

    def test_blade_setter_by_name(self):
        ch, fake = _make_chopper({"blade=6": ""})
        ch.blade = "MC1F10HP"
        assert b"blade=6\r" in fake.writes

    def test_reference_setter_by_name(self):
        ch, fake = _make_chopper({"ref=3": ""})
        ch.reference = "EXT-INNER"
        assert b"ref=3\r" in fake.writes

    def test_query_raises_on_command_error(self):
        # Simulate firmware rejecting a fractional phase (real v4.19 behaviour).
        ch, _fake = _make_chopper({"phase=19.5": "Command Error: CMD_ARG_INVALID"})
        with pytest.raises(RuntimeError, match="rejected"):
            ch.query("phase=19.5")

    def test_configure_pump_master_command_sequence(self):
        ch, fake = _make_chopper({
            "blade=6": "",
            "ref=1": "",
            "output=1": "",
            "freq=250": "",
            "enable=1": "",
        })
        ch.configure_pump_master(blade="MC1F10HP", frequency_hz=250)
        joined = b"".join(fake.writes)
        for sub in (b"blade=6\r", b"ref=1\r", b"output=1\r",
                    b"freq=250\r", b"enable=1\r"):
            assert sub in joined, f"missing command {sub!r}"

    def test_configure_probe_slave_command_sequence(self):
        ch, fake = _make_chopper({
            "blade=6": "",
            "ref=3": "",
            "nharmonic=1": "",
            "dharmonic=2": "",
            "phase=20": "",
            "enable=1": "",
        })
        ch.configure_probe_slave(
            blade="MC1F10HP", n_harmonic=1, d_harmonic=2, phase_deg=20,
        )
        joined = b"".join(fake.writes)
        for sub in (b"blade=6\r", b"ref=3\r", b"nharmonic=1\r",
                    b"dharmonic=2\r", b"phase=20\r", b"enable=1\r"):
            assert sub in joined, f"missing command {sub!r}"

    def test_close_closes_serial(self):
        ch, fake = _make_chopper({})
        ch.close()
        assert fake.is_open is False
        assert ch.is_open is False


# ---------------------------------------------------------------------------
# ChopperPair
# ---------------------------------------------------------------------------


class TestChopperPair:
    def test_setup_pair_applies_master_then_slave(self):
        pump = MockMC2000B(port="COM_PUMP")
        probe = MockMC2000B(port="COM_PROBE")
        pair = ChopperPair(pump=pump, probe=probe)
        pair.setup_pair(phase_deg=20)

        assert pump.reference == REF_INPUT_HIGHPREC["INT-INNER"]
        assert pump.enabled is True
        assert probe.reference == REF_INPUT_HIGHPREC["EXT-INNER"]
        assert probe.phase_deg == 20
        assert probe.enabled is True

    def test_shutdown_disables_and_closes(self):
        pump = MockMC2000B()
        probe = MockMC2000B()
        pump.enabled = True
        probe.enabled = True
        pair = ChopperPair(pump=pump, probe=probe)
        pair.shutdown()
        assert pump.enabled is False
        assert probe.enabled is False

    def test_pair_handles_none_probe(self):
        pump = MockMC2000B()
        pair = ChopperPair(pump=pump, probe=None)
        pair.setup_pair()  # should not raise
        assert pump.enabled is True
        pair.shutdown()  # should not raise
        assert pump.enabled is False
