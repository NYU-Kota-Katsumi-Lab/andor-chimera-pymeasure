"""Transient Absorption scan engine.

``TransientAbsorptionEngine`` orchestrates the full TA pump-probe scan loop.
It runs in a ``QThread`` so the GUI stays responsive, and communicates back
to the main thread via Qt signals.

Pause/resume uses ``threading.Event`` (set = running, clear = paused).
Abort uses a separate ``threading.Event`` (set = abort requested).

Signals
-------
scan_started(int)
    Emitted when a new scan begins. Argument is the scan index.
point_started(int, float)
    Emitted before acquiring each delay point. Arguments: scan_idx, delay_ps.
point_completed(int, float)
    Emitted after each delay point. Arguments: scan_idx, delay_ps.
scan_completed()
    Emitted when all scans finish successfully.
aborted()
    Emitted when the scan is aborted.
error(str)
    Emitted on unhandled exception. Argument is the error message.
signal_updated(float, object, object)
    Emitted with (delay_ps, wavelengths, delta_signal) after each point.
map_updated(object, object, object)
    Emitted with (delays, wavelengths, delta_signal_matrix) after each point.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Optional

import numpy as np
from PySide6.QtCore import QObject, Signal

from andor_qt.ta.engine_base import _EngineBase

from andor_qt.ta.acquisition import (
    AcquisitionSession,
    acquire_static_at_delay,
)
from andor_qt.ta.scan_config import TAScanConfig, SPEED_OF_LIGHT_MM_PS, ps_to_um

log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())


def _validate_dark_shape(
    dark: Optional[np.ndarray], expected_length: Optional[int]
) -> tuple[Optional[np.ndarray], Optional[str]]:
    """Validate that a dark frame matches the expected spectrum length.

    A dark whose length differs from the binned spectrum cannot be broadcast
    against acquired frames, and the engine's per-point try/except would
    silently log + skip every point. This helper catches the mismatch up
    front so it can be reported once and the scan can proceed without dark
    subtraction.

    Args:
        dark: Stored dark spectrum (1-D array) or ``None``.
        expected_length: Expected spectrum length (camera xpixels // hbin),
            or ``None`` if not determinable.

    Returns:
        ``(dark_or_none, warning_message_or_none)``. When the shape is
        valid (or there is no dark), ``warning_message_or_none`` is ``None``.
        On mismatch, the returned dark is ``None`` and the message explains
        what went wrong and how to fix it.
    """
    if dark is None:
        return None, None
    dark_arr = np.asarray(dark)
    if expected_length is None or expected_length <= 0:
        # Can't validate — pass through unchanged.
        return dark_arr, None
    if dark_arr.ndim != 1 or dark_arr.shape[0] != expected_length:
        msg = (
            f"Dark frame shape {tuple(dark_arr.shape)} does not match the "
            f"current spectrum length ({expected_length},). The dark was "
            f"likely acquired with a different hbin. Proceeding without "
            f"dark subtraction — re-acquire the dark with the current "
            f"camera settings to enable it."
        )
        return None, msg
    return dark_arr, None


def _format_eta(
    elapsed_s: float, completed: int, remaining: int,
    est_per_pt_s: float = 0.0,
) -> str:
    """Format estimated time remaining as a human-readable string.

    Args:
        elapsed_s: Time elapsed so far.
        completed: Number of points completed.
        remaining: Number of points remaining.
        est_per_pt_s: Pre-computed estimate of seconds per point (used when
            completed == 0 to show a prediction before any data arrives).
    """
    if completed == 0:
        if est_per_pt_s > 0:
            return "~" + _format_time(est_per_pt_s * remaining)
        return "..."
    per_pt = elapsed_s / completed
    eta_s = per_pt * remaining
    return _format_time(eta_s)


def _format_time(seconds: float) -> str:
    """Format seconds as a human-readable duration string."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    return f"{seconds/3600:.1f}h"


def _estimate_point_time_s(
    camera_settings: dict, n_averages: int, static: bool = False,
) -> float:
    """Estimate time per delay point from camera settings.

    Args:
        camera_settings: Dict with exposure_time, vs_speed, hs_speed, hbin,
            trigger_mode.
        n_averages: Number of frames (static) or pump/ref pairs (alternating).
        static: If True, each point acquires n_averages single frames
            (not pairs). If False, each point acquires 2 × n_averages
            frames (pump + ref alternation).

    Returns:
        Estimated seconds per point, or 0.0 if settings are missing.
    """
    from andor_qt.ta.acquisition import _compute_frame_period_s

    frame_period = _compute_frame_period_s(camera_settings)
    frames_per_point = n_averages if static else 2 * n_averages
    return frame_period * frames_per_point


def _make_scan_folder(base_dir: str, sample_name: str = "") -> Path:
    """Create a timestamped subfolder for a scan run.

    Returns:
        Path to the created subfolder (e.g. ``base_dir/2026-03-26_161500_sample/``).
    """
    from datetime import datetime
    ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    name = f"{ts}_{sample_name}" if sample_name else ts
    folder = Path(base_dir) / name
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def _save_spectrum_file(
    save_dir: Path,
    scan_idx: int,
    delay_ps: float,
    wavelengths: np.ndarray,
    delta_signal: np.ndarray,
) -> None:
    """Write one delta-signal spectrum to a tab-delimited text file.

    Filename encodes scan index and stage position in µm.

    Args:
        save_dir: Directory (Path) to write into.
        scan_idx: Zero-based scan index.
        delay_ps: Delay in picoseconds (used to compute stage position).
        wavelengths: Wavelength axis (nm).  May be empty.
        delta_signal: ΔI/I₀ spectrum array.
    """
    position_um = (delay_ps * SPEED_OF_LIGHT_MM_PS / 2.0) * 1000.0
    filename = f"scan{scan_idx:03d}_pos{position_um:+.1f}um.txt"
    filepath = Path(save_dir) / filename

    lines = []
    if len(wavelengths) == len(delta_signal):
        for wl, ds in zip(wavelengths, delta_signal):
            lines.append(f"{float(wl):.4f}\t{float(ds):.8e}")
    else:
        for ds in delta_signal:
            lines.append(f"\t{float(ds):.8e}")

    filepath.write_text("\n".join(lines), encoding="utf-8")


class _ScanWorker(QObject):
    """QObject that runs in a QThread and executes the scan loop."""

    # Status signals
    scan_started = Signal(int)
    point_started = Signal(int, float)
    point_completed = Signal(int, float)
    scan_completed = Signal()
    aborted = Signal()
    error = Signal(str)

    # Data signals
    signal_updated = Signal(float, object, object)
    map_updated = Signal(object, object, object)
    raw_pair_updated = Signal(object, object, int, int, int)  # pumped, ref, n_matched, n_discarded, n_frames
    user_prompt = Signal(str)  # ask user to do something (e.g. block pump)
    status_updated = Signal(str)

    def __init__(self):
        super().__init__()
        self._pause_event = threading.Event()
        self._pause_event.set()  # Start in running state
        self._abort_event = threading.Event()
        self._config: Optional[TAScanConfig] = None
        self._hw_manager = None
        self._writer = None

    def setup(self, config: TAScanConfig, hw_manager, writer, camera_settings=None,
              trigger_gen=None, phase_reader=None, dark=None) -> None:
        """Configure the worker before starting."""
        self._config = config
        self._hw_manager = hw_manager
        self._writer = writer
        self._camera_settings = camera_settings
        self._trigger_gen = trigger_gen
        self._phase_reader = phase_reader
        self._dark = dark
        self._abort_event.clear()
        self._pause_event.set()
        self._user_response = threading.Event()

    def user_confirmed(self):
        """Called from UI thread when user confirms a prompt."""
        self._user_response.set()

    def run(self) -> None:
        """Execute the scan loop. Called from QThread."""
        config = self._config
        hw = self._hw_manager
        writer = self._writer
        trigger_gen = self._trigger_gen
        phase_reader = self._phase_reader

        # Accumulated data for map updates
        all_delays = []
        all_signals = []  # list of 1-D arrays

        # Get current hbin for wavelength calibration
        hbin = (self._camera_settings or {}).get("hbin", 1)
        if isinstance(hbin, str):
            hbin = int(hbin.replace("x", ""))
        get_wl = getattr(hw, "get_wavelengths", None)
        if callable(get_wl):
            self._wavelengths = np.asarray(get_wl(hbin=hbin))
        else:
            self._wavelengths = np.array([])

        # Validate dark-frame shape against the binned spectrum length. A
        # mismatched dark (e.g. acquired at a different hbin) would broadcast-
        # fail on every point and silently skip the whole scan. Catch it once
        # here, warn loudly, and proceed without dark subtraction.
        expected_len: Optional[int] = None
        if isinstance(self._wavelengths, np.ndarray) and self._wavelengths.size:
            expected_len = int(self._wavelengths.shape[0])
        else:
            cam = getattr(hw, "camera", None)
            xpx = getattr(cam, "xpixels", None) if cam is not None else None
            if isinstance(xpx, int) and isinstance(hbin, int) and hbin > 0:
                expected_len = xpx // hbin
        self._dark, _dark_warning = _validate_dark_shape(self._dark, expected_len)
        if _dark_warning is not None:
            log.warning(_dark_warning)
            self.status_updated.emit("WARNING: " + _dark_warning)

        # Create timestamped subfolder for spectra if saving is enabled
        spectra_folder = None
        if config.save_spectra_dir:
            spectra_folder = _make_scan_folder(config.save_spectra_dir, config.sample_name)
            log.info(f"Saving spectra to: {spectra_folder}")

        # Get motion axis (used by both regular and static scans)
        mm = getattr(hw, "motion_manager", None)
        if mm is not None and hasattr(mm, "set_axis_hardware_index"):
            mm.set_axis_hardware_index("delay", config.stage_axis)
        axis = getattr(mm, "get_axis", lambda _: None)("delay") if mm is not None else None

        # Static ON/OFF: two full scans with user prompt between
        if config.acquisition_mode == "static_onoff":
            self._run_static_scan(config, hw, writer, spectra_folder, axis)
            return

        # Move stage to the first delay point before starting acquisition
        trigger_mode = (self._camera_settings or {}).get("trigger_mode", "internal")
        if config.delay_list:
            first_delay = config.ordered_delays(0)[0]
            if axis is not None:
                target_mm = getattr(axis, "t0_offset_mm", 0.0) + (first_delay * SPEED_OF_LIGHT_MM_PS) / 2
                cur_mm = getattr(axis, "position", float("nan"))
                log.info(f"Moving stage to initial delay {first_delay:.2f} ps ({target_mm:.3f} mm) before scan")
                self.status_updated.emit(
                    f"Moving to start — {ps_to_um(first_delay):.0f} µm  "
                    f"current: {cur_mm:.3f} mm → commanded: {target_mm:.3f} mm"
                )
                if hasattr(axis, "move_fast"):
                    axis.move_fast(target_mm)
                else:
                    axis.position_ps = first_delay

        # Start NI DAQ hardware tasks
        if trigger_gen is not None:
            trigger_gen.start()

        # For non-chopper modes, start phase reader before session (session
        # only starts the phase reader for chopper_2x2 via drain()).
        is_chopper = config.acquisition_mode == "chopper_2x2" and phase_reader is not None
        if not is_chopper and phase_reader is not None:
            phase_reader.start()

        # AcquisitionSession owns the camera lifecycle:
        # - chopper_2x2: camera starts once in __enter__, stops in __exit__
        # - other modes: session delegates to mode-specific functions
        session = AcquisitionSession(
            hw, config,
            camera_settings=self._camera_settings,
            phase_reader=phase_reader,
        )

        try:
            with session:
                for scan_idx in range(config.n_scans):
                    if self._abort_event.is_set():
                        self.aborted.emit()
                        return

                    self.scan_started.emit(scan_idx)

                    # Drain stale frames + tags before each scan to prevent
                    # leftover data from contaminating the first point
                    _drain_frames, _n_drain = hw.camera.get_buffered_frames()
                    if _n_drain > 0 and phase_reader is not None:
                        spf = getattr(config, "shots_per_frame", 2)
                        try:
                            phase_reader.read_tags(_n_drain * spf)
                        except Exception:
                            phase_reader.drain()

                    if writer is not None:
                        writer.begin_scan(scan_idx)

                    ordered = config.ordered_delays(scan_idx)
                    n_pts = len(ordered)
                    n_scans = config.n_scans

                    import time as _time
                    _scan_t0 = _time.perf_counter()
                    _pts_completed = 0
                    _est_per_pt = _estimate_point_time_s(
                        self._camera_settings or {}, config.n_averages
                    )

                    for pt_idx, delay_ps in enumerate(ordered):
                        # Check abort
                        if self._abort_event.is_set():
                            self.aborted.emit()
                            return

                        # Wait if paused
                        while not self._pause_event.is_set():
                            if self._abort_event.is_set():
                                self.aborted.emit()
                                return
                            self._pause_event.wait(timeout=0.1)

                        self.point_started.emit(scan_idx, delay_ps)

                        eta_str = _format_eta(
                            _time.perf_counter() - _scan_t0, _pts_completed, n_pts - pt_idx,
                            est_per_pt_s=_est_per_pt,
                        )

                        # Move stage to target delay (blocks until MD? reports done)
                        if axis is not None:
                            axis.position_ps = delay_ps

                        target_mm = (
                            getattr(axis, "t0_offset_mm", 0.0) + (delay_ps * SPEED_OF_LIGHT_MM_PS) / 2
                            if axis is not None else float("nan")
                        )
                        log.info(
                            f"Scan {scan_idx+1}/{n_scans} pt {pt_idx+1}/{n_pts}: "
                            f"moving to {delay_ps:.2f} ps ({target_mm:.3f} mm)  ETA: {eta_str}"
                        )

                        _pos_um = ps_to_um(delay_ps)

                        def _raw_cb(pumped, ref, n_matched, n_discarded, n_frames,
                                    _si=scan_idx, _pi=pt_idx, _ns=n_scans, _np=n_pts,
                                    _um=_pos_um, _eta=eta_str):
                            self.raw_pair_updated.emit(pumped, ref, n_matched, n_discarded, n_frames)
                            valid_pct = 100.0 * (2 * n_matched) / n_frames if n_frames > 0 else 0.0
                            self.status_updated.emit(
                                f"pt {_pi+1}/{_np}  {_um:.0f} µm  "
                                f"pairs: {n_matched}  discarded: {n_discarded}  "
                                f"({valid_pct:.0f}% valid)  ETA: {_eta}"
                            )

                        def _progress_cb(n_acc, n_tgt, elapsed,
                                         _pi=pt_idx, _np=n_pts, _um=_pos_um, _eta=eta_str):
                            pct = 100.0 * n_acc / n_tgt if n_tgt > 0 else 0.0
                            self.status_updated.emit(
                                f"pt {_pi+1}/{_np}  {_um:.0f} µm  "
                                f"pairs: {n_acc}/{n_tgt} ({pct:.0f}%)  "
                                f"elapsed: {elapsed:.0f}s  ETA: {_eta}"
                            )

                        # Skip-on-error: if one point fails, log and continue
                        try:
                            delta_signal = session.acquire_one_cycle(
                                dark=self._dark, raw_callback=_raw_cb,
                                progress_callback=_progress_cb,
                                abort_check=self._abort_event.is_set,
                            )
                        except Exception as exc:
                            log.warning(f"Point {pt_idx+1}/{n_pts} failed: {exc} — skipping")
                            self.status_updated.emit(
                                f"pt {pt_idx+1}/{n_pts} SKIPPED: {exc}"
                            )
                            _pts_completed += 1
                            self.point_completed.emit(scan_idx, delay_ps)
                            continue

                        # Auto-save: write to HDF5 after each point
                        if writer is not None:
                            from andor_qt.ta.acquisition import last_acquisition_stats as _stats
                            stage_um = (delay_ps * SPEED_OF_LIGHT_MM_PS / 2.0) * 1000.0
                            writer.write_point(
                                scan_idx, delay_ps, delta_signal,
                                stage_position_um=stage_um,
                                pump_spectrum=_stats.get("pump_mean"),
                                ref_spectrum=_stats.get("ref_mean"),
                            )

                        self.signal_updated.emit(delay_ps, self._wavelengths, delta_signal)

                        if spectra_folder is not None:
                            try:
                                _save_spectrum_file(
                                    spectra_folder, scan_idx, delay_ps,
                                    self._wavelengths, delta_signal,
                                )
                                from andor_qt.ta.acquisition import last_acquisition_stats
                                stats = last_acquisition_stats
                                if stats:
                                    position_um = (delay_ps * SPEED_OF_LIGHT_MM_PS / 2.0) * 1000.0
                                    from pathlib import Path as _P
                                    for suffix, data in [
                                        ("pump", stats.get("pump_mean")),
                                        ("ref", stats.get("ref_mean")),
                                        ("pump_std", stats.get("pump_std")),
                                        ("ref_std", stats.get("ref_std")),
                                    ]:
                                        if data is not None and len(self._wavelengths) == len(data):
                                            fn = f"scan{scan_idx:03d}_pos{position_um:+.1f}um_{suffix}.txt"
                                            lines = []
                                            for wl, d in zip(self._wavelengths, data):
                                                lines.append(f"{float(wl):.4f}\t{float(d):.8e}")
                                            _P(spectra_folder / fn).write_text("\n".join(lines), encoding="utf-8")
                            except Exception as exc:
                                log.warning(f"Failed to save spectrum file: {exc}")

                        # Update 2-D map
                        all_delays.append(delay_ps)
                        all_signals.append(delta_signal)
                        if len(all_signals) > 0:
                            signal_matrix = np.array(all_signals)
                            self.map_updated.emit(
                                np.array(all_delays), self._wavelengths, signal_matrix
                            )

                        _pts_completed += 1
                        self.point_completed.emit(scan_idx, delay_ps)

                self.scan_completed.emit()

        except Exception as exc:
            log.exception("TA engine error")
            self.error.emit(str(exc))

        finally:
            # Stop NI DAQ hardware tasks
            if phase_reader is not None:
                try:
                    phase_reader.stop()
                except Exception:
                    pass
            if trigger_gen is not None:
                try:
                    trigger_gen.stop()
                except Exception:
                    pass
            # Always restore to internal trigger after scan ends or aborts
            _apply = getattr(getattr(hw, "camera", None), "apply_camera_settings", None)
            if trigger_mode in ("external", "fast_external") and callable(_apply):
                try:
                    _apply({"trigger_mode": "internal"})
                except Exception:
                    pass

    def _run_static_scan(self, config, hw, writer, spectra_folder, axis):
        """Static ON/OFF scan: two full passes through delay list.

        Pass 1: Acquire at each delay with pump ON (long averaging).
        Prompt user to block pump.
        Pass 2: Acquire at each delay with pump OFF (same averaging).
        Compute delta-OD = -log10(pump_spectrum / ref_spectrum) at each delay.
        """
        import time as _time

        _apply = getattr(getattr(hw, "camera", None), "apply_camera_settings", None)
        if callable(_apply) and self._camera_settings:
            _apply(self._camera_settings)

        ordered = config.ordered_delays(0)
        n_pts = len(ordered)
        camera = hw.camera

        try:
            if writer is not None:
                writer.begin_scan(0)

            # --- Pass 1: Pump ON ---
            self.scan_started.emit(0)
            self.status_updated.emit("Static Pass 1: Pump ON — scanning all delays...")
            log.info(f"Static scan pass 1 (pump ON): {n_pts} delays, {config.n_averages} avg each")

            pump_spectra = {}  # delay_ps -> averaged spectrum
            _pass1_t0 = _time.perf_counter()
            _est_per_pt = _estimate_point_time_s(
                self._camera_settings or {}, config.n_averages, static=True,
            )

            for pt_idx, delay_ps in enumerate(ordered):
                if self._abort_event.is_set():
                    self.aborted.emit()
                    return

                self.point_started.emit(0, delay_ps)

                eta_str = _format_eta(
                    _time.perf_counter() - _pass1_t0, pt_idx, n_pts - pt_idx,
                    est_per_pt_s=_est_per_pt,
                )

                if axis is not None:
                    axis.position_ps = delay_ps

                try:
                    avg, std, n = acquire_static_at_delay(
                        hw, config.n_averages, self._abort_event,
                        dark=self._dark, camera_settings=self._camera_settings,
                    )
                except Exception as exc:
                    log.warning(f"Pass 1 pt {pt_idx+1} failed: {exc} — skipping")
                    self.point_completed.emit(0, delay_ps)
                    continue

                pump_spectra[delay_ps] = (avg, std, n)

                self.status_updated.emit(
                    f"Pass 1 (pump ON): pt {pt_idx+1}/{n_pts}  "
                    f"{ps_to_um(delay_ps):.0f} µm  ETA: {eta_str}"
                )
                self.raw_pair_updated.emit(avg, avg, 1, 0, 2)
                self.point_completed.emit(0, delay_ps)

            if self._abort_event.is_set():
                self.aborted.emit()
                return

            # --- Prompt user to block pump ---
            self.status_updated.emit("Block the pump beam, then click Continue")
            self.user_prompt.emit("Pass 1 (pump ON) complete.\n\nBlock the pump beam and click Continue.")
            self._user_response.clear()

            while not self._user_response.is_set():
                if self._abort_event.is_set():
                    self.aborted.emit()
                    return
                _time.sleep(0.1)

            if self._abort_event.is_set():
                self.aborted.emit()
                return

            # --- Pass 2: Pump OFF ---
            self.scan_started.emit(1)
            self.status_updated.emit("Static Pass 2: Pump OFF — scanning all delays...")
            log.info(f"Static scan pass 2 (pump OFF): {n_pts} delays, {config.n_averages} avg each")

            all_delays = []
            all_signals = []
            _pass2_t0 = _time.perf_counter()

            for pt_idx, delay_ps in enumerate(ordered):
                if self._abort_event.is_set():
                    self.aborted.emit()
                    return

                self.point_started.emit(1, delay_ps)

                eta_str = _format_eta(
                    _time.perf_counter() - _pass2_t0, pt_idx, n_pts - pt_idx,
                    est_per_pt_s=_est_per_pt,
                )

                if delay_ps not in pump_spectra:
                    log.warning(f"Pass 2: no pump data for {delay_ps:.2f} ps — skipping")
                    self.point_completed.emit(1, delay_ps)
                    continue

                if axis is not None:
                    axis.position_ps = delay_ps

                try:
                    ref_avg, ref_std, ref_n = acquire_static_at_delay(
                        hw, config.n_averages, self._abort_event,
                        dark=self._dark, camera_settings=self._camera_settings,
                    )
                except Exception as exc:
                    log.warning(f"Pass 2 pt {pt_idx+1} failed: {exc} — skipping")
                    self.point_completed.emit(1, delay_ps)
                    continue
                pump_avg, pump_std, pump_n = pump_spectra[delay_ps]
                ref_safe = np.where(ref_avg == 0, 1.0, ref_avg)
                delta_od = -np.log10(pump_avg / ref_safe)

                self.signal_updated.emit(delay_ps, self._wavelengths, delta_od)
                self.raw_pair_updated.emit(pump_avg, ref_avg, 1, 0, 2)

                # HDF5 save
                if writer is not None:
                    stage_um = (delay_ps * SPEED_OF_LIGHT_MM_PS / 2.0) * 1000.0
                    writer.write_point(0, delay_ps, delta_od, stage_position_um=stage_um)

                # Save ref spectrum
                if spectra_folder is not None:
                    try:
                        position_um = (delay_ps * SPEED_OF_LIGHT_MM_PS / 2.0) * 1000.0
                        # Save pump, ref, and delta_od as separate files
                        from pathlib import Path as _P
                        for suffix, data in [("pump", pump_avg), ("ref", ref_avg), ("deltaOD", delta_od)]:
                            fn = f"pos{position_um:+.1f}um_{suffix}.txt"
                            lines = []
                            if len(self._wavelengths) == len(data):
                                for wl, d in zip(self._wavelengths, data):
                                    lines.append(f"{float(wl):.4f}\t{float(d):.8e}")
                            _P(spectra_folder / fn).write_text("\n".join(lines), encoding="utf-8")
                    except Exception as exc:
                        log.warning(f"Failed to save spectra: {exc}")

                all_delays.append(delay_ps)
                all_signals.append(delta_od)
                if len(all_signals) > 0:
                    self.map_updated.emit(
                        np.array(all_delays), self._wavelengths, np.array(all_signals)
                    )

                self.status_updated.emit(
                    f"Pass 2 (pump OFF): pt {pt_idx+1}/{n_pts}  {ps_to_um(delay_ps):.0f} µm"
                )
                self.point_completed.emit(1, delay_ps)

            log.info("Static scan complete")
            self.scan_completed.emit()

        except Exception as exc:
            log.exception("Static scan error")
            self.error.emit(str(exc))
        finally:
            # Restore internal trigger so the camera is usable after scan
            _restore = getattr(camera, "apply_camera_settings", None)
            if callable(_restore):
                try:
                    _restore({"trigger_mode": "internal"})
                except Exception:
                    pass

class TransientAbsorptionEngine(_EngineBase):
    """High-level engine for TA pump-probe scanning.

    Manages a QThread and exposes signals mirroring ``_ScanWorker``.
    """

    # Re-export signals from worker
    scan_started = Signal(int)
    point_started = Signal(int, float)
    point_completed = Signal(int, float)
    scan_completed = Signal()
    aborted = Signal()
    error = Signal(str)
    signal_updated = Signal(float, object, object)
    map_updated = Signal(object, object, object)
    raw_pair_updated = Signal(object, object, int, int, int)
    status_updated = Signal(str)
    user_prompt = Signal(str)

    def __init__(self, parent=None):
        worker = _ScanWorker()
        super().__init__(
            worker,
            [worker.scan_completed, worker.aborted, worker.error],
            parent,
        )

        # Forward worker signals
        worker.scan_started.connect(self.scan_started)
        worker.point_started.connect(self.point_started)
        worker.point_completed.connect(self.point_completed)
        worker.scan_completed.connect(self.scan_completed)
        worker.aborted.connect(self.aborted)
        worker.error.connect(self.error)
        worker.user_prompt.connect(self.user_prompt)
        worker.signal_updated.connect(self.signal_updated)
        worker.map_updated.connect(self.map_updated)
        worker.raw_pair_updated.connect(self.raw_pair_updated)
        worker.status_updated.connect(self.status_updated)

    def start_scan(self, config: TAScanConfig, hw_manager, writer=None,
                   camera_settings=None, trigger_gen=None, phase_reader=None,
                   dark=None) -> None:
        """Start the TA scan in a background thread."""
        if self.is_running:
            log.warning("Scan already running")
            return
        self._worker.setup(config, hw_manager, writer, camera_settings=camera_settings,
                           trigger_gen=trigger_gen, phase_reader=phase_reader,
                           dark=dark)
        self._start_worker()

    def pause(self) -> None:
        """Pause the scan after the current delay point completes."""
        self._worker._pause_event.clear()

    def resume(self) -> None:
        """Resume a paused scan."""
        self._worker._pause_event.set()

    def abort(self) -> None:
        """Abort the scan as soon as possible."""
        self._worker._abort_event.set()
        self._worker._pause_event.set()  # Unblock if paused
        self._abort_worker()

    def emergency_stop(self) -> None:
        """Abort scan and stop all motion."""
        self.abort()

