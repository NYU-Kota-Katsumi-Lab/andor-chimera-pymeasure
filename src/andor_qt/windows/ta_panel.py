"""TAWindowPanel — composite TA panel for main window integration.

Composes:
- ``TAScanConfigWidget`` (left) — delay list builder + scan parameters
- ``TALiveDisplayWidget`` (right) — real-time ΔOD plots
- Optional ``StageControlWidget`` (if a delay axis is available)

Wires ``scan_requested`` → ``TransientAbsorptionEngine.start_scan``.
Wires engine data signals → live display slots.
"""

from __future__ import annotations

import logging
import os
from typing import Optional, Tuple

import numpy as np
from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QScrollArea, QSplitter, QVBoxLayout, QWidget

from andor_qt.ta.engine import TransientAbsorptionEngine
from andor_qt.ta.hdf5_writer import TADataWriter, auto_filename
from andor_qt.ta.monitor_engine import TAMonitorEngine
from andor_qt.ta.scan_config import TAScanConfig
from andor_qt.widgets.hardware.chopper_control import ChopperControlWidget
from andor_qt.widgets.ta.live_display import TALiveDisplayWidget
from andor_qt.widgets.ta.monitor_widget import TAMonitorWidget
from andor_qt.widgets.ta.scan_config_widget import TAScanConfigWidget

log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())


def _make_daq_hardware(config: TAScanConfig) -> Tuple:
    """Create trigger generator and phase reader for the configured acquisition mode.

    Returns ``(trigger_gen, phase_reader)`` — either or both may be ``None``
    depending on the mode. Uses mock objects when ``ANDOR_MOCK=1``.
    """
    trigger_gen = None
    phase_reader = None

    if config.acquisition_mode == "chopper_2x2":
        if os.environ.get("ANDOR_MOCK"):
            from andor_qt.ta.nidaq_phase import MockNIDAQChopper2x2Reader
            from andor_qt.ta.nidaq_trigger import MockNIDAQChopper500Hz
            trigger_gen = MockNIDAQChopper500Hz()
            phase_reader = MockNIDAQChopper2x2Reader()
        else:
            from andor_qt.ta.nidaq_phase import NIDAQPhaseReader
            from andor_qt.ta.nidaq_trigger import NIDAQChopper500Hz
            trigger_gen = NIDAQChopper500Hz(
                device=config.nidaq_device,
                clock_source=config.nidaq_clock_source,
                sync_source=config.nidaq_chopper_sync_source,
                counter=config.nidaq_chopper_counter,
            )
            phase_reader = NIDAQPhaseReader(
                device=config.nidaq_device,
                di_channel=config.nidaq_di_channel,
                clock_source=config.nidaq_clock_source,
                clock_rate=config.nidaq_clock_rate,
            )
        if config.external_trigger:
            log.info("External camera trigger — NIDAQChopper500Hz not started")
            trigger_gen = None

    elif config.acquisition_mode == "shot_to_shot":
        log.info("shot_to_shot mode — camera triggered by PFI0 at 1 kHz")
        if os.environ.get("ANDOR_MOCK"):
            from andor_qt.ta.nidaq_phase import MockNIDAQPhaseReader
            phase_reader = MockNIDAQPhaseReader()
        else:
            from andor_qt.ta.nidaq_phase import NIDAQPhaseReader
            phase_reader = NIDAQPhaseReader(
                device=config.nidaq_device,
                di_channel=config.nidaq_di_channel,
                clock_source=config.nidaq_clock_source,
                clock_rate=config.nidaq_clock_rate,
            )

    return trigger_gen, phase_reader


class TAWindowPanel(QWidget):
    """TA measurement panel composed of config, engine, and live display.

    Signals:
        camera_busy: Emitted with ``True`` when the TA module starts using
            the camera (scan or monitor) and ``False`` when it finishes.
            The main window connects this to disable/re-enable the
            spectrometer acquire and queue controls.

    Args:
        hw_manager: Hardware manager instance.
        parent: Optional parent widget.
    """

    camera_busy = Signal(bool)

    def __init__(self, hw_manager, parent=None):
        super().__init__(parent)
        self._hw_manager = hw_manager

        # --- Engine ---
        self._engine = TransientAbsorptionEngine(self)
        self._writer: Optional[TADataWriter] = None
        self._current_config: Optional[TAScanConfig] = None
        self._dark_frame: Optional[np.ndarray] = None
        self._dark_acquiring: bool = False

        # In-memory buffer for post-scan save (always populated, regardless
        # of whether HDF5 auto-save is enabled)
        self._last_scan_buffer: dict = {
            "delays_ps": [],
            "delta_signals": [],
            "pump_spectra": [],
            "ref_spectra": [],
            "stage_positions_um": [],
            "wavelengths": None,
            "config": None,
            "camera_settings": None,
        }

        # --- Trigger test state ---
        self._trigger_test_running = False
        self._trigger_test_gen = None

        # --- Monitor engine ---
        self._monitor_engine = TAMonitorEngine(self)

        # --- Widgets ---
        self._config_widget = TAScanConfigWidget()
        self._config_widget.set_hardware_manager(hw_manager)
        self._monitor_widget = TAMonitorWidget(hw_manager)
        self._live_display = TALiveDisplayWidget()

        # --- Left pane: tabs for Scan / Monitor ---
        from PySide6.QtWidgets import QTabWidget
        self._left_tabs = QTabWidget()
        self._left_tabs.addTab(self._config_widget, "Scan")
        self._left_tabs.addTab(self._monitor_widget, "Monitor")

        # --- Chopper control widget ---
        chopper_pair = getattr(hw_manager, "chopper_pair", None)
        self._chopper_widget = ChopperControlWidget(chopper_pair=chopper_pair)

        # --- Status / trigger-test bar ---
        self._status_label = QLabel("Ready")
        self._status_label.setStyleSheet("padding: 2px 6px; font-size: 12px;")
        self._trigger_test_btn = QPushButton("Start Trigger Test")
        self._trigger_test_btn.setFixedWidth(150)
        self._trigger_test_btn.setToolTip(
            "Start NIDAQChopper500Hz without a scan so you can probe CTR1OUT on an oscilloscope"
        )
        self._trigger_test_btn.clicked.connect(self._toggle_trigger_test)

        self._save_last_btn = QPushButton("Save HDF5...")
        self._save_last_btn.setFixedWidth(110)
        self._save_last_btn.setToolTip(
            "Save the most recent scan to HDF5 (works even if auto-save was off)"
        )
        self._save_last_btn.setEnabled(False)
        self._save_last_btn.clicked.connect(self._on_save_last_scan)

        self._save_spectra_btn = QPushButton("Save Spectra...")
        self._save_spectra_btn.setFixedWidth(120)
        self._save_spectra_btn.setToolTip(
            "Save per-point spectrum text files (works even if auto-save was off)"
        )
        self._save_spectra_btn.setEnabled(False)
        self._save_spectra_btn.clicked.connect(self._on_save_last_spectra)

        status_row = QHBoxLayout()
        status_row.addWidget(self._status_label, stretch=1)
        status_row.addWidget(self._save_last_btn, stretch=0)
        status_row.addWidget(self._save_spectra_btn, stretch=0)
        status_row.addWidget(self._trigger_test_btn, stretch=0)

        # --- Layout ---
        live_scroll = QScrollArea()
        live_scroll.setWidget(self._live_display)
        live_scroll.setWidgetResizable(True)
        live_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        # Pack tabs + chopper widget into a single left container so the
        # splitter still treats the left side as one column.
        left_container = QWidget()
        left_layout = QVBoxLayout(left_container)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.addWidget(self._left_tabs, stretch=1)
        left_layout.addWidget(self._chopper_widget, stretch=0)

        splitter = QSplitter()
        splitter.addWidget(left_container)
        splitter.addWidget(live_scroll)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)
        splitter.setSizes([700, 900])

        layout = QVBoxLayout(self)
        layout.addWidget(splitter, stretch=1)
        layout.addLayout(status_row, stretch=0)

        # --- Signal wiring: Scan ---
        self._config_widget.scan_requested.connect(self._on_scan_requested)
        self._config_widget.abort_requested.connect(self._engine.abort)
        self._config_widget._external_trigger_check.toggled.connect(
            lambda checked: self._trigger_test_btn.setEnabled(not checked)
        )

        # Engine → live display
        self._engine.signal_updated.connect(self._live_display.on_signal_updated)
        self._engine.map_updated.connect(self._live_display.on_map_updated)
        self._engine.raw_pair_updated.connect(self._live_display.on_raw_pair_updated)

        # Engine → in-memory buffer for post-scan save
        self._engine.signal_updated.connect(self._on_buffer_signal_updated)
        self._engine.raw_pair_updated.connect(self._on_buffer_raw_pair_updated)

        # --- Signal wiring: Monitor ---
        self._monitor_widget.monitor_requested.connect(self._on_monitor_requested)
        self._monitor_widget.stop_requested.connect(self._on_monitor_stop)
        self._monitor_widget.static_acquire_requested.connect(self._on_static_acquire_requested)
        self._monitor_widget.dark_requested.connect(self._on_dark_requested)
        self._monitor_widget.dark_cleared.connect(self._on_dark_cleared)

        # Static ON/OFF state
        self._static_pump_avg = None
        self._static_ref_avg = None
        self._monitor_engine.cycle_completed.connect(self._on_monitor_cycle)
        self._monitor_engine.raw_pair_updated.connect(self._live_display.on_raw_pair_updated)
        self._monitor_engine.raw_pair_updated.connect(self._cache_monitor_raw)
        self._engine.raw_pair_updated.connect(self._cache_monitor_raw)
        self._monitor_engine.status_updated.connect(self._status_label.setText)
        self._monitor_engine.stopped.connect(self._on_monitor_stopped)
        self._monitor_engine.error.connect(self._on_monitor_error)
        self._monitor_engine.user_prompt.connect(self._on_user_prompt)
        self._monitor_engine.static_completed.connect(self._on_static_completed)
        self._monitor_engine.single_phase_completed.connect(self._on_single_phase_completed)

        # camera_busy → chopper widget: disable writes during scan/monitor
        self.camera_busy.connect(self._chopper_widget.set_busy)

        # Re-bind the chopper widget once HardwareManager finishes its
        # background init (panel is built before chopper_pair exists).
        from andor_qt.core.signals import get_hardware_signals
        get_hardware_signals().choppers_initialized.connect(self._on_choppers_initialized)
        # In case init already completed before this connection was made,
        # do an opportunistic bind now.
        if getattr(self._hw_manager, "chopper_pair", None) is not None:
            self._chopper_widget.set_pair(self._hw_manager.chopper_pair)

        # Engine → status / button state
        self._engine.scan_started.connect(self._on_scan_started)
        self._engine.point_started.connect(lambda _si, _d: self._live_display.reset_phase_stats())
        self._engine.point_completed.connect(self._on_point_completed)
        self._engine.scan_completed.connect(self._on_scan_completed)
        self._engine.aborted.connect(self._on_aborted)
        self._engine.error.connect(self._on_engine_error)
        self._engine.user_prompt.connect(self._on_engine_user_prompt)

        self._engine.status_updated.connect(self._status_label.setText)

    @Slot(object)
    def _on_scan_requested(self, config: TAScanConfig) -> None:
        """Start scan when config widget emits scan_requested."""
        log.info(f"TA scan requested: {config.sample_name}, "
                 f"{len(config.delay_list)} delays, {config.n_scans} scans")
        self._current_config = config
        self._config_widget.set_scan_running(True)
        self.camera_busy.emit(True)
        self._status_label.setText(
            f"Starting scan — {len(config.delay_list)} delays × {config.n_scans} scan(s)"
        )
        self._live_display.clear()
        camera_settings = self._config_widget.camera_settings
        self._pre_set_wavelengths(camera_settings)

        # Reset post-scan save buffer for new scan
        self._reset_scan_buffer(config, camera_settings)
        self._save_last_btn.setEnabled(False)
        self._save_spectra_btn.setEnabled(False)

        # Create HDF5 writer if a save directory is configured
        writer: Optional[TADataWriter] = None
        if config.save_hdf5_dir:
            try:
                hbin = camera_settings.get("hbin", 1)
                if isinstance(hbin, str):
                    hbin = int(hbin.replace("x", ""))
                get_wl = getattr(self._hw_manager, "get_wavelengths", None)
                wavelengths = np.asarray(get_wl(hbin=hbin)) if callable(get_wl) else np.array([])
                h5_path = auto_filename(
                    config.sample_name or "ta_scan", config.save_hdf5_dir
                )
                hardware_info = self._collect_hardware_info(config)
                writer = TADataWriter(
                    h5_path, wavelengths=wavelengths,
                    sample_name=config.sample_name, notes=config.notes,
                    scan_config=config,
                    camera_settings=camera_settings,
                    hardware_info=hardware_info,
                )
                writer.open()
                self._writer = writer
                log.info(f"HDF5 writer opened: {h5_path}")
            except Exception as exc:
                log.error(f"Failed to create HDF5 writer: {exc}")
                writer = None
                self._writer = None

        trigger_gen, phase_reader = _make_daq_hardware(config)

        self._engine.start_scan(config, self._hw_manager, writer=writer,
                                 camera_settings=camera_settings,
                                 trigger_gen=trigger_gen,
                                 phase_reader=phase_reader,
                                 dark=self._dark_frame)

    def _reset_scan_buffer(self, config: TAScanConfig, camera_settings: dict) -> None:
        """Clear the in-memory scan buffer at the start of a new scan."""
        self._last_scan_buffer = {
            "delays_ps": [],
            "delta_signals": [],
            "pump_spectra": [],
            "ref_spectra": [],
            "stage_positions_um": [],
            "wavelengths": None,
            "config": config,
            "camera_settings": dict(camera_settings) if camera_settings else None,
        }

    @Slot(float, object, object)
    def _on_buffer_signal_updated(self, delay_ps: float, wavelengths, delta_signal) -> None:
        """Append a delta signal to the post-scan save buffer."""
        import numpy as np
        from andor_pymeasure.instruments.motion_controller import SPEED_OF_LIGHT_MM_PS

        buf = self._last_scan_buffer
        buf["delays_ps"].append(float(delay_ps))
        buf["delta_signals"].append(np.asarray(delta_signal, dtype=np.float64).copy())
        buf["stage_positions_um"].append((delay_ps * SPEED_OF_LIGHT_MM_PS / 2.0) * 1000.0)
        if buf["wavelengths"] is None and wavelengths is not None and len(wavelengths) > 0:
            buf["wavelengths"] = np.asarray(wavelengths, dtype=np.float64).copy()

    @Slot(object, object, int, int, int)
    def _on_buffer_raw_pair_updated(self, pumped, ref, n_matched: int,
                                     n_discarded: int, n_frames: int) -> None:
        """Capture pump/ref spectra into the buffer at the same cadence as delta signals."""
        import numpy as np
        buf = self._last_scan_buffer
        # raw_pair_updated fires once per acquisition cycle, in lockstep with signal_updated
        buf["pump_spectra"].append(np.asarray(pumped, dtype=np.float64).copy())
        buf["ref_spectra"].append(np.asarray(ref, dtype=np.float64).copy())

    def _on_save_last_scan(self) -> None:
        """Prompt for a file path and save the in-memory scan buffer to HDF5."""
        from PySide6.QtWidgets import QFileDialog, QMessageBox
        import numpy as np

        buf = self._last_scan_buffer
        if not buf["delays_ps"]:
            QMessageBox.information(self, "Save Last Scan", "No scan data in memory.")
            return

        config = buf["config"]
        default_name = (config.sample_name if config else "ta_scan") or "ta_scan"
        path_str, _ = QFileDialog.getSaveFileName(
            self,
            "Save Last Scan",
            f"{default_name}.h5",
            "HDF5 files (*.h5);;All files (*)",
        )
        if not path_str:
            return

        from pathlib import Path
        path = Path(path_str)
        if path.suffix == "":
            path = path.with_suffix(".h5")

        try:
            wavelengths = buf["wavelengths"]
            if wavelengths is None:
                wavelengths = np.array([])
            hardware_info = self._collect_hardware_info(config) if config else None
            with TADataWriter(
                path, wavelengths=wavelengths,
                sample_name=(config.sample_name if config else ""),
                notes=(config.notes if config else ""),
                scan_config=config,
                camera_settings=buf["camera_settings"],
                hardware_info=hardware_info,
            ) as writer:
                writer.begin_scan(0)
                # Pump/ref may be shorter than delays if some points failed; pad with None
                n = len(buf["delays_ps"])
                pumps = buf["pump_spectra"]
                refs = buf["ref_spectra"]
                positions = buf["stage_positions_um"]
                for i in range(n):
                    pump_i = pumps[i] if i < len(pumps) else None
                    ref_i = refs[i] if i < len(refs) else None
                    writer.write_point(
                        scan_idx=0,
                        delay_ps=buf["delays_ps"][i],
                        delta_signal=buf["delta_signals"][i],
                        stage_position_um=positions[i],
                        pump_spectrum=pump_i,
                        ref_spectrum=ref_i,
                    )
            log.info(f"Post-scan save complete: {path}")
            QMessageBox.information(
                self, "Save Last Scan",
                f"Saved {n} points to:\n{path}"
            )
        except Exception as exc:
            log.exception("Post-scan save failed")
            QMessageBox.critical(
                self, "Save Failed", f"Could not save scan:\n{exc}"
            )

    def _on_save_last_spectra(self) -> None:
        """Prompt for a parent directory and save per-point spectrum text files.

        Mirrors the auto-save format used during scans: creates a timestamped
        subfolder containing one ``scanNNN_pos±X.Xum.txt`` per delay point,
        plus pump/ref text files when available.
        """
        from PySide6.QtWidgets import QFileDialog, QMessageBox
        from pathlib import Path
        from andor_qt.ta.engine import _save_spectrum_file, _make_scan_folder
        from andor_pymeasure.instruments.motion_controller import SPEED_OF_LIGHT_MM_PS

        buf = self._last_scan_buffer
        if not buf["delays_ps"]:
            QMessageBox.information(self, "Save Spectra", "No scan data in memory.")
            return

        parent_str = QFileDialog.getExistingDirectory(
            self, "Choose Parent Directory for Spectra Folder"
        )
        if not parent_str:
            return

        config = buf["config"]
        sample_name = (config.sample_name if config else "") or ""
        try:
            spectra_folder = _make_scan_folder(parent_str, sample_name)
            wavelengths = buf["wavelengths"]
            n = len(buf["delays_ps"])
            n_pumps = len(buf["pump_spectra"])
            n_refs = len(buf["ref_spectra"])
            for i in range(n):
                delay_ps = buf["delays_ps"][i]
                delta = buf["delta_signals"][i]
                _save_spectrum_file(
                    spectra_folder, scan_idx=0, delay_ps=delay_ps,
                    wavelengths=wavelengths if wavelengths is not None else [],
                    delta_signal=delta,
                )
                # Save pump and ref spectra alongside (matches engine auto-save)
                position_um = (delay_ps * SPEED_OF_LIGHT_MM_PS / 2.0) * 1000.0
                for suffix, data in [
                    ("pump", buf["pump_spectra"][i] if i < n_pumps else None),
                    ("ref", buf["ref_spectra"][i] if i < n_refs else None),
                ]:
                    if data is None:
                        continue
                    fn = f"scan000_pos{position_um:+.1f}um_{suffix}.txt"
                    lines = []
                    if wavelengths is not None and len(wavelengths) == len(data):
                        for wl, d in zip(wavelengths, data):
                            lines.append(f"{float(wl):.4f}\t{float(d):.8e}")
                    else:
                        for d in data:
                            lines.append(f"\t{float(d):.8e}")
                    (spectra_folder / fn).write_text("\n".join(lines), encoding="utf-8")

            log.info(f"Post-scan spectra save complete: {spectra_folder}")
            QMessageBox.information(
                self, "Save Spectra",
                f"Saved {n} spectra to:\n{spectra_folder}"
            )
        except Exception as exc:
            log.exception("Post-scan spectra save failed")
            QMessageBox.critical(
                self, "Save Failed", f"Could not save spectra:\n{exc}"
            )

    def _collect_hardware_info(self, config: TAScanConfig) -> dict:
        """Collect hardware metadata for HDF5 file."""
        hw = self._hw_manager
        info: dict = {}

        # Camera info
        try:
            cam = hw.camera
            if cam is not None:
                info["camera_serial"] = str(getattr(cam, "_serial", ""))
                info["camera_model"] = "DU970P Newton EMCCD"
                temp = getattr(cam, "temperature", None)
                if temp is not None:
                    info["camera_temperature_c"] = float(temp)
                # EM gain mode tells analysts whether the em_gain integer in
                # /metadata is a physical multiplication factor (mode 3) or
                # an opaque DAC code (mode 0/1).
                em_mode = getattr(cam, "em_gain_mode", None)
                if em_mode is not None:
                    info["em_gain_mode"] = int(em_mode)
                get_em = getattr(cam, "get_em_gain", None)
                if callable(get_em):
                    info["em_gain_actual"] = int(get_em())
        except Exception:
            pass

        # Spectrograph info
        try:
            sp = hw.spectrograph
            if sp is not None:
                if sp.info is not None:
                    info["spectrograph_serial"] = sp.info.serial_number
                    info["num_gratings"] = sp.info.num_gratings
                grating_idx = sp.grating
                info["grating_index"] = grating_idx
                for g in (sp.info.gratings if sp.info else ()):
                    if g.index == grating_idx:
                        info["grating_lines_per_mm"] = g.lines_per_mm
                        info["grating_blaze"] = g.blaze
                        break
                info["center_wavelength_nm"] = sp.wavelength
        except Exception:
            pass

        # Stage info
        info["stage_axis"] = config.stage_axis
        try:
            mm = hw.motion_manager
            if mm is not None:
                axis = mm.get_axis("delay")
                if axis is not None:
                    info["stage_axis_hw_index"] = axis.index
                    info["stage_t0_offset_mm"] = getattr(axis, "t0_offset_mm", 0.0)
        except Exception:
            pass

        return info

    def _finalize_writer(self) -> None:
        if self._writer is not None:
            try:
                self._writer.finalize()
            except Exception as exc:
                log.warning(f"Error finalizing HDF5 writer: {exc}")
            self._writer = None

    def _end_scan(self) -> None:
        """Common cleanup after any scan termination."""
        self._finalize_writer()
        self._config_widget.set_scan_running(False)
        self._trigger_test_btn.setEnabled(True)
        self._current_config = None
        self.camera_busy.emit(False)

    @Slot(int)
    def _on_scan_started(self, scan_idx: int) -> None:
        # Stop trigger test if running before scan takes over the counter
        if self._trigger_test_running:
            self._toggle_trigger_test()
        n_scans = self._current_config.n_scans if self._current_config else "?"
        self._status_label.setText(f"Scan {scan_idx + 1}/{n_scans} — starting…")
        self._trigger_test_btn.setEnabled(False)

    @Slot(int, float)
    def _on_point_completed(self, scan_idx: int, delay_ps: float) -> None:
        pass  # status_updated signal provides richer detail; keep slot for future use

    @Slot()
    def _on_scan_completed(self) -> None:
        log.info("TA scan completed")
        self._status_label.setText("Scan complete")
        # Enable post-scan save if any data was collected
        if len(self._last_scan_buffer["delays_ps"]) > 0:
            self._save_last_btn.setEnabled(True)
            self._save_spectra_btn.setEnabled(True)
        self._end_scan()

    @Slot()
    def _on_aborted(self) -> None:
        log.info("TA scan aborted")
        self._status_label.setText("Scan aborted")
        # Allow saving partial data
        if len(self._last_scan_buffer["delays_ps"]) > 0:
            self._save_last_btn.setEnabled(True)
            self._save_spectra_btn.setEnabled(True)
        self._end_scan()

    @Slot(str)
    def _on_engine_error(self, message: str) -> None:
        log.error(f"TA engine error: {message}")
        self._status_label.setText(f"Error: {message}")
        self._end_scan()

    @Slot(dict)
    def _on_choppers_initialized(self, info: dict) -> None:
        """Re-bind the chopper widget once HardwareManager finishes init."""
        pair = getattr(self._hw_manager, "chopper_pair", None)
        self._chopper_widget.set_pair(pair)
        if pair is not None:
            self._chopper_widget.refresh()
        log.info(
            "Chopper widget re-bound (pump=%s connected=%s, probe=%s connected=%s)",
            info.get("pump_port", ""), info.get("pump_connected", False),
            info.get("probe_port", ""), info.get("probe_connected", False),
        )

    @Slot(str)
    def _on_engine_user_prompt(self, msg: str) -> None:
        """Handle user prompt from engine (e.g. static_onoff pump block)."""
        from PySide6.QtWidgets import QMessageBox
        result = QMessageBox.information(
            self, "Action Required", msg,
            QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel,
        )
        if result == QMessageBox.StandardButton.Ok:
            self._engine.user_confirmed()
        else:
            self._engine.abort()

    def _toggle_trigger_test(self) -> None:
        """Start or stop NIDAQChopper500Hz without a scan for oscilloscope testing."""
        if self._trigger_test_running:
            # Stop
            if self._trigger_test_gen is not None:
                try:
                    self._trigger_test_gen.stop()
                except Exception as exc:
                    log.warning(f"Trigger test stop error: {exc}")
                self._trigger_test_gen = None
            self._trigger_test_running = False
            self._trigger_test_btn.setText("Start Trigger Test")
            self._status_label.setText("Trigger test stopped")
            log.info("Trigger test stopped")
        else:
            # Start
            config = self._config_widget._build_config()
            if os.environ.get("ANDOR_MOCK"):
                from andor_qt.ta.nidaq_trigger import MockNIDAQChopper500Hz
                gen = MockNIDAQChopper500Hz()
            else:
                from andor_qt.ta.nidaq_trigger import NIDAQChopper500Hz
                gen = NIDAQChopper500Hz(
                    device=config.nidaq_device,
                    clock_source=config.nidaq_clock_source,
                    sync_source=config.nidaq_chopper_sync_source,
                    counter=config.nidaq_chopper_counter,
                )
            try:
                gen.start()
                self._trigger_test_gen = gen
                self._trigger_test_running = True
                self._trigger_test_btn.setText("Stop Trigger Test")
                self._status_label.setText(
                    "Trigger test RUNNING — probe CTR1OUT (PFI13) on oscilloscope"
                )
                log.info("Trigger test started")
            except Exception as exc:
                log.error(f"Trigger test start failed: {exc}")
                self._status_label.setText(f"Trigger test failed: {exc}")

    # -- Monitor mode ------------------------------------------------------

    def _pre_set_wavelengths(self, camera_settings: dict) -> None:
        """Pre-set wavelengths on live display so raw spectra save with nm axis."""
        hbin = camera_settings.get("hbin", 1)
        if isinstance(hbin, str):
            hbin = int(hbin.replace("x", ""))
        get_wl = getattr(self._hw_manager, "get_wavelengths", None)
        if callable(get_wl):
            self._live_display._wavelengths = np.asarray(get_wl(hbin=hbin))

    @Slot(object)
    def _on_monitor_requested(self, config: TAScanConfig) -> None:
        """Start monitor mode."""
        if self._monitor_engine.is_running:
            return

        # Stop trigger test if running
        if self._trigger_test_running:
            self._toggle_trigger_test()

        camera_settings = self._monitor_widget.camera_settings
        trigger_gen, phase_reader = _make_daq_hardware(config)

        self._live_display.set_monitor_mode(True)
        self._live_display.clear()
        self._monitor_widget.set_monitor_running(True)
        self._config_widget.set_scan_running(True)  # lock out scan
        self.camera_busy.emit(True)
        self._status_label.setText("Monitor starting...")
        self._pre_set_wavelengths(camera_settings)

        self._monitor_engine.start_monitor(
            config, self._hw_manager,
            camera_settings=camera_settings,
            trigger_gen=trigger_gen,
            phase_reader=phase_reader,
            dark=self._dark_frame,
        )

    def _on_monitor_cycle(self, wavelengths, delta, avg_delta) -> None:
        """Handle monitor cycle completion — update live display."""
        # Dark frame acquisition: store raw average and stop after first cycle
        if getattr(self, "_dark_acquiring", False):
            self._dark_acquiring = False
            # Use the pump-ON average as the dark frame (shutter should be closed)
            if hasattr(self._monitor_widget, "_last_pump"):
                self._dark_frame = np.asarray(self._monitor_widget._last_pump).copy()
                from datetime import datetime
                ts = datetime.now().strftime("%H:%M:%S")
                n = getattr(self._monitor_widget, "_last_n_on", 0)
                self._monitor_widget.set_dark_status(f"Dark: {n} frames, {ts}")
                self._status_label.setText(f"Dark frame acquired ({n} frames)")
                log.info(f"Dark frame stored: {len(self._dark_frame)} pixels, {n} frames")
            self._monitor_engine.stop()
            return

        # Normal monitor cycle
        axis = None
        if self._hw_manager.motion_manager:
            axis = self._hw_manager.motion_manager.get_axis("delay")
        delay_ps = getattr(axis, "position_ps", 0.0) if axis else 0.0
        self._live_display.on_signal_updated(delay_ps, wavelengths, delta)
        self._monitor_widget.update_position()

    def _cache_monitor_raw(self, pumped, ref, n_matched, n_discarded, n_frames):
        """Cache the last raw pair for the monitor save buttons."""
        wl = (
            self._live_display._wavelengths
            if len(self._live_display._wavelengths) == len(pumped)
            else None
        )
        self._monitor_widget.cache_raw_data(pumped, ref, n_matched, n_matched, wl)

    @Slot()
    def _on_monitor_stop(self) -> None:
        """Stop monitor mode."""
        self._monitor_engine.stop()

    @Slot()
    def _on_monitor_stopped(self) -> None:
        """Handle monitor engine stopped."""
        self._monitor_widget.set_monitor_running(False)
        self._config_widget.set_scan_running(False)
        self._live_display.set_monitor_mode(False)
        self.camera_busy.emit(False)
        self._status_label.setText("Monitor stopped")

    @Slot(str)
    def _on_monitor_error(self, msg: str) -> None:
        """Handle monitor engine error."""
        self._monitor_widget.set_monitor_running(False)
        self._config_widget.set_scan_running(False)
        self._live_display.set_monitor_mode(False)
        self.camera_busy.emit(False)
        self._status_label.setText(f"Monitor error: {msg}")
        log.error(f"Monitor error: {msg}")

    # -- Dark frame -----------------------------------------------------------

    def _on_dark_requested(self, config: TAScanConfig) -> None:
        """Acquire dark frame using the monitor engine in single-cycle mode."""
        if self._monitor_engine.is_running:
            return

        camera_settings = self._monitor_widget.camera_settings
        trigger_gen, phase_reader = _make_daq_hardware(config)

        self._monitor_widget.set_monitor_running(True)
        self._config_widget.set_scan_running(True)
        self.camera_busy.emit(True)
        self._status_label.setText("Acquiring dark frame...")
        self._pre_set_wavelengths(camera_settings)

        # Use single cycle — the cycle_completed callback stores the result
        self._dark_acquiring = True
        self._monitor_engine.start_monitor(
            config, self._hw_manager,
            camera_settings=camera_settings,
            trigger_gen=trigger_gen,
            phase_reader=phase_reader,
        )

    def _on_dark_cleared(self) -> None:
        """Clear the stored dark frame."""
        self._dark_frame = None
        self._status_label.setText("Dark frame cleared")
        log.info("Dark frame cleared")

    @Slot(str)
    def _on_user_prompt(self, msg: str) -> None:
        """Show a dialog asking user to do something (e.g. block pump)."""
        from PySide6.QtWidgets import QMessageBox
        result = QMessageBox.information(
            self, "Action Required", msg,
            QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel,
        )
        if result == QMessageBox.StandardButton.Ok:
            self._monitor_engine.user_confirmed()
        else:
            self._monitor_engine.stop()

    @Slot(object, object, object, object)
    def _on_static_completed(self, wavelengths, pump_avg, ref_avg, delta_od) -> None:
        """Handle static ON/OFF completion — display ΔOD in signal plot."""
        self._live_display.on_signal_updated(0.0, wavelengths, delta_od)
        self._live_display.on_raw_pair_updated(
            pump_avg, ref_avg, 1, 0, 2
        )

    @Slot(str, object)
    def _on_static_acquire_requested(self, phase: str, config: TAScanConfig) -> None:
        """Handle static single-phase acquisition (pump ON or pump OFF)."""
        if self._monitor_engine.is_running:
            return

        camera_settings = self._monitor_widget.camera_settings

        self._monitor_widget.set_monitor_running(True)
        self._config_widget.set_scan_running(True)
        self.camera_busy.emit(True)
        self._status_label.setText(f"Static: acquiring {phase}...")
        self._pre_set_wavelengths(camera_settings)

        self._monitor_engine.start_monitor(
            config, self._hw_manager,
            camera_settings=camera_settings,
            static_phase=phase,
        )

    @Slot(str, object, object)
    def _on_single_phase_completed(self, phase: str, wavelengths, avg_spectrum) -> None:
        """Store result of a static single-phase acquisition."""
        from datetime import datetime
        ts = datetime.now().strftime("%H:%M:%S")

        if phase == "pump":
            self._static_pump_avg = avg_spectrum
            self._static_pump_time = ts
            self._static_wavelengths = wavelengths
            log.info(f"Static pump ON collected at {ts}: mean={avg_spectrum.mean():.1f}")
        else:
            self._static_ref_avg = avg_spectrum
            self._static_ref_time = ts
            self._static_wavelengths = wavelengths
            log.info(f"Static pump OFF collected at {ts}: mean={avg_spectrum.mean():.1f}")

        # Update status with timestamps
        pump_done = self._static_pump_avg is not None
        ref_done = self._static_ref_avg is not None
        self._monitor_widget.update_static_status(
            pump_done, ref_done,
            pump_time=getattr(self, "_static_pump_time", ""),
            ref_time=getattr(self, "_static_ref_time", ""),
        )

        # Show raw spectra — use cached pump/ref so both curves display correctly
        pump = self._static_pump_avg if self._static_pump_avg is not None else avg_spectrum
        ref = self._static_ref_avg if self._static_ref_avg is not None else avg_spectrum
        self._live_display.on_raw_pair_updated(pump, ref, 1, 0, 2)

        # Update save cache so "Save Pump ON"/"Save Pump OFF" buttons use correct data
        self._monitor_widget.cache_raw_data(pump, ref, 1, 1, wavelengths)

        # If both phases collected, compute ΔOD
        if pump_done and ref_done:
            ref_safe = np.where(self._static_ref_avg == 0, 1.0, self._static_ref_avg)
            delta_od = -np.log10(self._static_pump_avg / ref_safe)
            self._live_display.on_signal_updated(0.0, self._static_wavelengths, delta_od)
            self._status_label.setText(
                f"Static ON/OFF complete  |  "
                f"\u0394OD range: [{delta_od.min():.6f}, {delta_od.max():.6f}]"
            )
            log.info(f"Static ΔOD computed: [{delta_od.min():.6f}, {delta_od.max():.6f}]")

    # -- public API --------------------------------------------------------

    @property
    def engine(self) -> TransientAbsorptionEngine:
        return self._engine

    @property
    def config_widget(self) -> TAScanConfigWidget:
        return self._config_widget

    @property
    def monitor_widget(self) -> TAMonitorWidget:
        return self._monitor_widget

    @property
    def live_display(self) -> TALiveDisplayWidget:
        return self._live_display

    @property
    def chopper_widget(self) -> ChopperControlWidget:
        return self._chopper_widget
