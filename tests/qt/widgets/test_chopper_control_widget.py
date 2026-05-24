"""Tests for ChopperControlWidget."""

from __future__ import annotations

import pytest
from PySide6.QtCore import Qt

from andor_pymeasure.instruments.thorlabs_mc2000b import (
    BLADE_INDEX,
    REF_INPUT_HIGHPREC,
    ChopperPair,
    MockMC2000B,
)
from andor_qt.widgets.hardware.chopper_control import ChopperControlWidget


@pytest.fixture
def pair():
    """Pump + probe ``MockMC2000B`` pair, both bound to a ChopperPair."""
    pump = MockMC2000B(port="COM_PUMP")
    probe = MockMC2000B(port="COM_PROBE")
    return ChopperPair(pump=pump, probe=probe)


class TestChopperControlWidgetInit:
    def test_construct_with_none_disables_buttons(self, qt_app):
        w = ChopperControlWidget(chopper_pair=None)
        assert w.setup_button.isEnabled() is False
        assert w.refresh_button.isEnabled() is False

    def test_construct_with_pair_shows_ports(self, qt_app, pair):
        w = ChopperControlWidget(chopper_pair=pair)
        assert "COM_PUMP" in w.pump_panel._port_label.text()
        assert "COM_PROBE" in w.probe_panel._port_label.text()

    def test_construct_with_pair_enables_setup(self, qt_app, pair):
        w = ChopperControlWidget(chopper_pair=pair)
        assert w.setup_button.isEnabled() is True
        assert w.refresh_button.isEnabled() is True


class TestChopperControlWidgetReadback:
    def test_refresh_pulls_phase_from_chopper(self, qt_app, pair):
        pair.pump.phase_deg = 42
        pair.probe.phase_deg = 19
        w = ChopperControlWidget(chopper_pair=pair)
        w.refresh()
        assert w.pump_panel._phase_spin.value() == 42
        assert w.probe_panel._phase_spin.value() == 19

    def test_refresh_pulls_enabled_state(self, qt_app, pair):
        pair.pump.enabled = True
        pair.probe.enabled = False
        w = ChopperControlWidget(chopper_pair=pair)
        w.refresh()
        assert w.pump_panel._enable_check.isChecked() is True
        assert w.probe_panel._enable_check.isChecked() is False

    def test_refresh_shows_actual_frequency(self, qt_app, pair):
        pair.pump.frequency_hz = 250
        pair.pump.n_harmonic = 1
        pair.pump.d_harmonic = 1
        w = ChopperControlWidget(chopper_pair=pair)
        w.refresh()
        assert "250" in w.pump_panel._freq_label.text()


class TestChopperControlWidgetWrite:
    def test_phase_apply_writes_to_chopper(self, qt_app, pair):
        w = ChopperControlWidget(chopper_pair=pair)
        w.probe_panel._phase_spin.setValue(123)
        w.probe_panel._on_phase_committed()
        assert pair.probe.phase_deg == 123

    def test_phase_signal_emitted(self, qt_app, pair):
        w = ChopperControlWidget(chopper_pair=pair)
        captured = []
        w.probe_phase_changed.connect(captured.append)
        w.probe_panel._phase_spin.setValue(77)
        w.probe_panel._on_phase_committed()
        assert captured == [77]

    def test_enable_toggle_writes_to_chopper(self, qt_app, pair):
        w = ChopperControlWidget(chopper_pair=pair)
        # Programmatically toggle by clicking
        w.pump_panel._enable_check.setChecked(True)
        assert pair.pump.enabled is True
        w.pump_panel._enable_check.setChecked(False)
        assert pair.pump.enabled is False


class TestChopperControlWidgetHarmonic:
    def test_refresh_pulls_harmonic_from_chopper(self, qt_app, pair):
        pair.pump.n_harmonic = 1
        pair.pump.d_harmonic = 2
        pair.probe.n_harmonic = 3
        pair.probe.d_harmonic = 1
        w = ChopperControlWidget(chopper_pair=pair)
        w.refresh()
        assert w.pump_panel._n_spin.value() == 1
        assert w.pump_panel._d_spin.value() == 2
        assert w.probe_panel._n_spin.value() == 3
        assert w.probe_panel._d_spin.value() == 1

    def test_harmonic_apply_writes_to_chopper(self, qt_app, pair):
        w = ChopperControlWidget(chopper_pair=pair)
        w.pump_panel._n_spin.setValue(2)
        w.pump_panel._d_spin.setValue(3)
        w.pump_panel._on_harmonic_committed()
        assert pair.pump.n_harmonic == 2
        assert pair.pump.d_harmonic == 3

    def test_harmonic_signal_emitted(self, qt_app, pair):
        w = ChopperControlWidget(chopper_pair=pair)
        captured = []
        w.pump_harmonic_changed.connect(lambda n, d: captured.append((n, d)))
        w.pump_panel._n_spin.setValue(1)
        w.pump_panel._d_spin.setValue(2)
        w.pump_panel._on_harmonic_committed()
        assert captured == [(1, 2)]

    def test_harmonic_apply_button_clickable(self, qt_app, pair):
        w = ChopperControlWidget(chopper_pair=pair)
        w.probe_panel._n_spin.setValue(1)
        w.probe_panel._d_spin.setValue(1)
        w.probe_panel._harmonic_apply.click()
        assert pair.probe.n_harmonic == 1
        assert pair.probe.d_harmonic == 1


class TestChopperControlWidgetSetup:
    def test_setup_button_applies_pair_config(self, qt_app, pair):
        w = ChopperControlWidget(chopper_pair=pair)
        # Pre-set a probe phase the user wants to preserve through setup.
        w.probe_panel._phase_spin.setValue(20)
        w.setup_button.click()
        assert pair.pump.reference == REF_INPUT_HIGHPREC["INT-INNER"]
        assert pair.pump.enabled is True
        assert pair.probe.reference == REF_INPUT_HIGHPREC["EXT-INNER"]
        assert pair.probe.enabled is True
        assert pair.probe.phase_deg == 20
        assert pair.pump.blade == BLADE_INDEX["MC1F10HP"]

    def test_setup_emits_signal_with_options(self, qt_app, pair):
        w = ChopperControlWidget(chopper_pair=pair)
        captured = []
        w.setup_requested.connect(captured.append)
        w.setup_button.click()
        assert len(captured) == 1
        assert captured[0]["blade"] == "MC1F10HP"
        assert captured[0]["frequency_hz"] == 250


class TestChopperControlWidgetBusy:
    def test_set_busy_disables_writes(self, qt_app, pair):
        w = ChopperControlWidget(chopper_pair=pair)
        w.set_busy(True)
        assert w.pump_panel._phase_spin.isEnabled() is False
        assert w.probe_panel._enable_check.isEnabled() is False
        assert w.pump_panel._n_spin.isEnabled() is False
        assert w.pump_panel._d_spin.isEnabled() is False
        assert w.pump_panel._harmonic_apply.isEnabled() is False
        assert w.setup_button.isEnabled() is False

    def test_set_busy_leaves_refresh_enabled(self, qt_app, pair):
        w = ChopperControlWidget(chopper_pair=pair)
        w.set_busy(True)
        # Refresh is read-only; still usable while a scan is running.
        assert w.refresh_button.isEnabled() is True

    def test_set_busy_false_re_enables(self, qt_app, pair):
        w = ChopperControlWidget(chopper_pair=pair)
        w.set_busy(True)
        w.set_busy(False)
        assert w.pump_panel._phase_spin.isEnabled() is True
        assert w.setup_button.isEnabled() is True


class TestChopperControlWidgetRebinding:
    def test_set_pair_none_disables(self, qt_app, pair):
        w = ChopperControlWidget(chopper_pair=pair)
        assert w.setup_button.isEnabled() is True
        w.set_pair(None)
        assert w.setup_button.isEnabled() is False
        assert w.refresh_button.isEnabled() is False

    def test_set_pair_then_back(self, qt_app, pair):
        w = ChopperControlWidget(chopper_pair=None)
        assert w.setup_button.isEnabled() is False
        w.set_pair(pair)
        assert w.setup_button.isEnabled() is True


class TestChopperControlWidgetSingleChopper:
    """Edge case: probe port not connected (only pump present)."""

    def test_probe_panel_disabled_when_probe_is_none(self, qt_app):
        pump = MockMC2000B(port="COM_PUMP")
        only_pump = ChopperPair(pump=pump, probe=None)
        w = ChopperControlWidget(chopper_pair=only_pump)
        assert "COM_PUMP" in w.pump_panel._port_label.text()
        assert "not connected" in w.probe_panel._port_label.text()
        assert w.probe_panel._phase_spin.isEnabled() is False
        # Setup button still usable — it tolerates None probe
        assert w.setup_button.isEnabled() is True
