"""ChopperControlWidget — Thorlabs MC2000B pair control panel.

Displays per-chopper enable, phase, and frequency readbacks for the
pump master (COM4) and probe slave (COM3). Provides a "Setup pair" button
that applies the canonical pump-master / probe-slave configuration.

The widget reads/writes the underlying ``MC2000B`` objects on demand. It
does NOT poll automatically — call :meth:`refresh` after init and after
serial-bound operations. This avoids spamming the serial line while a TA
scan is running. The widget is disabled while a scan is in progress (the
TA panel calls :meth:`set_busy(True)`).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

if TYPE_CHECKING:
    from andor_pymeasure.instruments.thorlabs_mc2000b import (
        ChopperPair,
        MC2000B,
        MockMC2000B,
    )

log = logging.getLogger(__name__)


class _SingleChopperPanel(QGroupBox):
    """Per-chopper sub-panel: enable checkbox, phase spinbox, freq label."""

    enable_changed = Signal(bool)
    phase_committed = Signal(int)

    def __init__(self, title: str, parent: Optional[QWidget] = None):
        super().__init__(title, parent)
        self._chopper = None  # type: Optional["MC2000B"]
        self._build_ui()

    def _build_ui(self) -> None:
        form = QFormLayout(self)
        form.setLabelAlignment(Qt.AlignRight)
        form.setSpacing(6)

        self._port_label = QLabel("—")
        self._port_label.setStyleSheet("font-family: monospace;")
        form.addRow("Port:", self._port_label)

        self._enable_check = QCheckBox("Enabled")
        self._enable_check.toggled.connect(self._on_enable_toggled)
        form.addRow("Motor:", self._enable_check)

        phase_row = QHBoxLayout()
        self._phase_spin = QSpinBox()
        self._phase_spin.setRange(0, 359)
        self._phase_spin.setSingleStep(1)
        self._phase_spin.setSuffix("°")
        self._phase_spin.setToolTip(
            "Phase in degrees. MC2000B firmware v4.19 only accepts integer "
            "phases 0–359."
        )
        # Commit on editing-finished (Enter or focus-out) so spinner clicks
        # don't fire a write per tick.
        self._phase_spin.editingFinished.connect(self._on_phase_committed)
        phase_row.addWidget(self._phase_spin)
        self._phase_apply = QPushButton("Apply")
        self._phase_apply.setMaximumWidth(70)
        self._phase_apply.clicked.connect(self._on_phase_committed)
        phase_row.addWidget(self._phase_apply)
        form.addRow("Phase:", phase_row)

        self._freq_label = QLabel("—")
        self._freq_label.setStyleSheet("font-family: monospace;")
        form.addRow("Blade freq:", self._freq_label)

        self._status_label = QLabel("")
        self._status_label.setWordWrap(True)
        self._status_label.setStyleSheet("color: #666; font-size: 11px;")
        form.addRow(self._status_label)

    def bind(self, chopper) -> None:
        """Bind a chopper object to this panel. ``None`` means not connected."""
        self._chopper = chopper
        if chopper is None:
            self._port_label.setText("<not connected>")
            self._enable_check.setEnabled(False)
            self._phase_spin.setEnabled(False)
            self._phase_apply.setEnabled(False)
            self._freq_label.setText("—")
            self._status_label.setText("")
            return
        self._port_label.setText(str(chopper.port))
        self._enable_check.setEnabled(True)
        self._phase_spin.setEnabled(True)
        self._phase_apply.setEnabled(True)
        self.refresh()

    def refresh(self) -> None:
        """Read live values from the chopper and update the UI."""
        if self._chopper is None:
            return
        try:
            enabled = self._chopper.enabled
            self._enable_check.blockSignals(True)
            self._enable_check.setChecked(enabled)
            self._enable_check.blockSignals(False)
        except Exception as e:
            self._status_label.setText(f"Read enable failed: {e}")
            return

        try:
            phase = self._chopper.phase_deg
            self._phase_spin.blockSignals(True)
            self._phase_spin.setValue(phase)
            self._phase_spin.blockSignals(False)
        except Exception as e:
            self._status_label.setText(f"Read phase failed: {e}")

        try:
            actual = self._chopper.actual_frequency_hz
            self._freq_label.setText(f"{actual:.2f} Hz")
        except Exception as e:
            self._freq_label.setText(f"err: {e}")

        self._status_label.setText("")

    def set_busy(self, busy: bool) -> None:
        """Disable all interactive controls while ``busy``."""
        self._enable_check.setEnabled(not busy and self._chopper is not None)
        self._phase_spin.setEnabled(not busy and self._chopper is not None)
        self._phase_apply.setEnabled(not busy and self._chopper is not None)

    # -- internal slots ----------------------------------------------------

    def _on_enable_toggled(self, checked: bool) -> None:
        if self._chopper is None:
            return
        try:
            self._chopper.enabled = checked
            self.enable_changed.emit(checked)
            self._status_label.setText(
                "Enabled" if checked else "Disabled"
            )
        except Exception as e:
            log.error("Toggle enable on %s failed: %s", self._chopper.port, e)
            self._status_label.setText(f"Enable failed: {e}")
            self._enable_check.blockSignals(True)
            self._enable_check.setChecked(not checked)
            self._enable_check.blockSignals(False)

    def _on_phase_committed(self) -> None:
        if self._chopper is None:
            return
        target = int(self._phase_spin.value())
        try:
            self._chopper.phase_deg = target
            actual = self._chopper.phase_deg
            self.phase_committed.emit(actual)
            self._status_label.setText(f"Phase set to {actual}°")
        except Exception as e:
            log.error("Set phase on %s failed: %s", self._chopper.port, e)
            self._status_label.setText(f"Set phase failed: {e}")


class ChopperControlWidget(QGroupBox):
    """Qt widget for an MC2000B chopper pair.

    Args:
        chopper_pair: :class:`andor_pymeasure.instruments.thorlabs_mc2000b.ChopperPair`
            (or None to disable the widget).
        parent: Optional Qt parent.

    Signals:
        setup_requested: emitted when the user clicks "Setup pair"; payload
            is the dict of options sent to ``ChopperPair.setup_pair()``.
        pump_phase_changed(int): emitted with the new pump phase in degrees.
        probe_phase_changed(int): emitted with the new probe phase in
            degrees.
    """

    setup_requested = Signal(dict)
    pump_phase_changed = Signal(int)
    probe_phase_changed = Signal(int)

    def __init__(
        self,
        chopper_pair: Optional["ChopperPair"] = None,
        parent: Optional[QWidget] = None,
    ):
        super().__init__("Choppers (Thorlabs MC2000B)", parent)
        self._pair: Optional["ChopperPair"] = None
        self._build_ui()
        self.set_pair(chopper_pair)

    # -- UI construction ---------------------------------------------------

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)

        grid = QGridLayout()
        self._pump_panel = _SingleChopperPanel("Pump (master)")
        self._probe_panel = _SingleChopperPanel("Probe (slave)")
        grid.addWidget(self._pump_panel, 0, 0)
        grid.addWidget(self._probe_panel, 0, 1)
        root.addLayout(grid)

        self._pump_panel.phase_committed.connect(self.pump_phase_changed)
        self._probe_panel.phase_committed.connect(self.probe_phase_changed)

        controls = QHBoxLayout()

        self._refresh_btn = QPushButton("Refresh")
        self._refresh_btn.setToolTip("Re-read enable/phase/freq from both choppers")
        self._refresh_btn.clicked.connect(self.refresh)
        controls.addWidget(self._refresh_btn)

        self._setup_btn = QPushButton("Setup pair")
        self._setup_btn.setToolTip(
            "Apply canonical pump-master + probe-slave configuration "
            "(blade=MC1F10HP, 250 Hz, N/D=1/1, phase=current probe phase)"
        )
        self._setup_btn.clicked.connect(self._on_setup_clicked)
        controls.addWidget(self._setup_btn)

        controls.addStretch()
        root.addLayout(controls)

    # -- public API --------------------------------------------------------

    def set_pair(self, pair: Optional["ChopperPair"]) -> None:
        """Bind a chopper pair to this widget (or unbind with ``None``)."""
        self._pair = pair
        if pair is None:
            self._pump_panel.bind(None)
            self._probe_panel.bind(None)
            self._setup_btn.setEnabled(False)
            self._refresh_btn.setEnabled(False)
            return
        self._pump_panel.bind(pair.pump)
        self._probe_panel.bind(pair.probe)
        self._setup_btn.setEnabled(True)
        self._refresh_btn.setEnabled(True)

    def refresh(self) -> None:
        """Re-read all parameters from both choppers."""
        self._pump_panel.refresh()
        self._probe_panel.refresh()

    def set_busy(self, busy: bool) -> None:
        """Disable interactive controls (used while a scan is running)."""
        self._pump_panel.set_busy(busy)
        self._probe_panel.set_busy(busy)
        self._setup_btn.setEnabled(not busy and self._pair is not None)
        # Refresh is read-only; allow it during scans
        self._refresh_btn.setEnabled(self._pair is not None)

    # -- accessors used by tests + integration -----------------------------

    @property
    def pump_panel(self) -> _SingleChopperPanel:
        return self._pump_panel

    @property
    def probe_panel(self) -> _SingleChopperPanel:
        return self._probe_panel

    @property
    def setup_button(self) -> QPushButton:
        return self._setup_btn

    @property
    def refresh_button(self) -> QPushButton:
        return self._refresh_btn

    # -- internal slots ----------------------------------------------------

    def _on_setup_clicked(self) -> None:
        if self._pair is None:
            return
        # Probe phase is the per-session tuning knob; preserve whatever the
        # user has typed in the probe phase spinbox at the moment of setup.
        try:
            probe_phase = int(self._probe_panel._phase_spin.value())
        except Exception:
            probe_phase = 0
        opts = {
            "blade": "MC1F10HP",
            "frequency_hz": 250,
            "n_harmonic": 1,
            "d_harmonic": 1,
            "phase_deg": probe_phase,
        }
        try:
            self._pair.setup_pair(**opts)
            self.setup_requested.emit(opts)
            self.refresh()
        except Exception as e:
            log.error("Chopper setup_pair failed: %s", e)
