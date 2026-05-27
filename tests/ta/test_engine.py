"""Tests for TransientAbsorptionEngine scan loop and acquire helpers.

Uses MockHardwareManager to avoid real hardware.
Engine runs in a QThread — tests use signal capture with a short timeout.
"""

from __future__ import annotations

import threading
import time
from typing import List
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from andor_qt.ta.scan_config import TAScanConfig
from andor_qt.ta.engine import (
    TransientAbsorptionEngine,
    _save_spectrum_file,
    _validate_dark_shape,
)
from andor_qt.ta.acquisition import acquire_delta_signal_at_delay


# ---------------------------------------------------------------------------
# Minimal mock hardware manager
# ---------------------------------------------------------------------------


def make_mock_hw(n_pixels: int = 64):
    """Create a minimal mock HardwareManager for TA engine tests."""
    hw = MagicMock()
    # Camera returns a flat spectrum
    hw.camera.get_spectrum.return_value = np.ones(n_pixels) * 1000.0
    hw.camera.get_image.return_value = np.ones((10, n_pixels)) * 1000.0
    # Batch read for chopper_2x2 mode
    hw.camera.start_run_till_abort.return_value = None
    hw.camera.get_buffered_frames.return_value = (
        np.ones((200, n_pixels)) * 1000.0, 200
    )
    hw.camera.abort_acquisition.return_value = None

    # Mock motion axis — set real float values so engine f-string formatting works
    mock_axis = MagicMock()
    mock_axis.t0_offset_mm = 0.0
    mock_axis.position = 0.0
    mock_axis.position_ps = 0.0
    mock_axis.position_mm = 0.0
    hw.motion.get_axis.return_value = mock_axis
    hw.motion_manager.get_axis.return_value = mock_axis
    hw.wavelengths = np.linspace(400.0, 800.0, n_pixels)
    return hw


def make_config(n_delays: int = 3, n_averages: int = 1, n_scans: int = 1):
    delays = [float(i) for i in range(n_delays)]
    return TAScanConfig(
        delay_list=delays,
        n_averages=n_averages,
        n_scans=n_scans,
        acquisition_mode="boxcar",
        scan_direction="forward",
        sample_name="test",
    )


# ---------------------------------------------------------------------------
# _save_spectrum_file — no headers
# ---------------------------------------------------------------------------


class TestSaveSpectrumFileNoHeaders:
    """Verify _save_spectrum_file writes data only, no comment lines."""

    def test_no_comment_lines(self, tmp_path):
        wl = np.linspace(400, 800, 8)
        sig = np.random.randn(8) * 1e-3
        _save_spectrum_file(tmp_path, scan_idx=0, delay_ps=1.5,
                            wavelengths=wl, delta_signal=sig)
        files = list(tmp_path.glob("*.txt"))
        assert len(files) == 1
        for line in files[0].read_text().splitlines():
            assert not line.startswith("#"), f"Comment found: {line!r}"

    def test_two_tab_columns(self, tmp_path):
        wl = np.linspace(400, 800, 4)
        sig = np.ones(4) * 0.001
        _save_spectrum_file(tmp_path, scan_idx=0, delay_ps=0.0,
                            wavelengths=wl, delta_signal=sig)
        files = list(tmp_path.glob("*.txt"))
        lines = files[0].read_text().splitlines()
        assert len(lines) == 4
        for line in lines:
            cols = line.split("\t")
            assert len(cols) == 2

    def test_line_count_matches_pixels(self, tmp_path):
        wl = np.linspace(400, 800, 16)
        sig = np.zeros(16)
        _save_spectrum_file(tmp_path, scan_idx=1, delay_ps=5.0,
                            wavelengths=wl, delta_signal=sig)
        files = list(tmp_path.glob("*.txt"))
        lines = files[0].read_text().splitlines()
        assert len(lines) == 16


# ---------------------------------------------------------------------------
# _validate_dark_shape
# ---------------------------------------------------------------------------


class TestValidateDarkShape:
    def test_none_dark_passes_through(self):
        dark, warn = _validate_dark_shape(None, 200)
        assert dark is None
        assert warn is None

    def test_matching_shape_returns_dark_unchanged(self):
        dark = np.arange(200, dtype=float)
        result, warn = _validate_dark_shape(dark, 200)
        assert np.array_equal(result, dark)
        assert warn is None

    def test_mismatched_shape_clears_dark_and_warns(self):
        dark = np.zeros(1600)  # hbin=1
        result, warn = _validate_dark_shape(dark, 200)  # scan at hbin=8
        assert result is None
        assert warn is not None
        assert "(1600,)" in warn
        assert "200" in warn
        assert "hbin" in warn.lower()

    def test_unknown_expected_length_passes_through(self):
        dark = np.zeros(1600)
        result, warn = _validate_dark_shape(dark, None)
        assert result is not None
        assert warn is None

    def test_two_d_dark_is_rejected(self):
        dark = np.zeros((10, 200))
        result, warn = _validate_dark_shape(dark, 200)
        assert result is None
        assert warn is not None


# ---------------------------------------------------------------------------
# acquire_delta_signal_at_delay
# ---------------------------------------------------------------------------


class TestAcquireDeltaSignalAtDelay:
    def test_returns_ndarray(self):
        hw = make_mock_hw()
        config = make_config(n_averages=2)
        result = acquire_delta_signal_at_delay(
            delay_ps=1.0, hw_manager=hw, config=config, dark=None
        )
        assert isinstance(result, np.ndarray)

    def test_result_length_matches_pixels(self):
        hw = make_mock_hw(n_pixels=32)
        config = make_config(n_averages=1)
        result = acquire_delta_signal_at_delay(0.0, hw, config, dark=None)
        assert len(result) == 32

    def test_dark_subtraction_applied(self):
        hw = make_mock_hw(n_pixels=10)
        # Camera always returns 1000, dark = 100 → net = 900
        hw.camera.get_spectrum.return_value = np.ones(10) * 1000.0
        config = make_config(n_averages=1)
        dark = np.ones(10) * 100.0
        result = acquire_delta_signal_at_delay(0.0, hw, config, dark=dark)
        # ΔI/I₀ of identical pumped/ref = 0
        assert result == pytest.approx(np.zeros(10), abs=1e-8)


# ---------------------------------------------------------------------------
# TransientAbsorptionEngine — signal capture helpers
# ---------------------------------------------------------------------------


def collect_signals(engine, signal_name: str, timeout: float = 5.0) -> List:
    """Collect emitted signal values into a list."""
    received = []
    sig = getattr(engine, signal_name)
    sig.connect(lambda *args: received.append(args))
    return received


class TestTransientAbsorptionEngineSignals:
    """Test that the engine emits expected signals during a scan."""

    def _run_engine(self, engine, config, hw, writer=None, timeout=10.0):
        """Start engine scan and wait for it to finish, processing Qt events."""
        import time as _time
        from PySide6.QtWidgets import QApplication

        done = [False]
        engine.scan_completed.connect(lambda: done.__setitem__(0, True))
        engine.aborted.connect(lambda: done.__setitem__(0, True))
        engine.error.connect(lambda msg: done.__setitem__(0, True))

        engine.start_scan(config, hw, writer)

        start = _time.time()
        while not done[0] and _time.time() - start < timeout:
            QApplication.instance().processEvents()
            _time.sleep(0.005)

    def test_scan_started_emitted(self, qt_app):
        hw = make_mock_hw()
        config = make_config(n_delays=2, n_scans=1)
        engine = TransientAbsorptionEngine()

        started = []
        engine.scan_started.connect(lambda idx: started.append(idx))

        self._run_engine(engine, config, hw)
        assert len(started) == 1
        assert started[0] == 0

    def test_scan_completed_emitted(self, qt_app):
        hw = make_mock_hw()
        config = make_config(n_delays=2, n_scans=1)
        engine = TransientAbsorptionEngine()

        done = []
        engine.scan_completed.connect(lambda: done.append(True))

        self._run_engine(engine, config, hw)
        assert len(done) == 1

    def test_point_completed_emitted_for_each_delay(self, qt_app):
        hw = make_mock_hw()
        config = make_config(n_delays=3, n_scans=1)
        engine = TransientAbsorptionEngine()

        points = []
        engine.point_completed.connect(lambda idx, delay: points.append((idx, delay)))

        self._run_engine(engine, config, hw)
        assert len(points) == 3

    def test_abort_stops_scan(self, qt_app):
        from PySide6.QtWidgets import QApplication

        hw = make_mock_hw()
        # Slow down acquisition so we can abort mid-scan
        hw.camera.get_spectrum.side_effect = lambda: (time.sleep(0.01), np.ones(64) * 1000.0)[1]
        config = make_config(n_delays=10, n_scans=5)
        engine = TransientAbsorptionEngine()

        aborted = []
        engine.aborted.connect(lambda: aborted.append(True))

        done = [False]
        engine.aborted.connect(lambda: done.__setitem__(0, True))
        engine.scan_completed.connect(lambda: done.__setitem__(0, True))

        engine.start_scan(config, hw, None)
        time.sleep(0.05)
        engine.abort()

        start = time.time()
        while not done[0] and time.time() - start < 5.0:
            QApplication.instance().processEvents()
            time.sleep(0.005)

        assert len(aborted) >= 1

    def test_multiple_scans_emits_scan_started_for_each(self, qt_app):
        hw = make_mock_hw()
        config = make_config(n_delays=2, n_scans=3)
        engine = TransientAbsorptionEngine()

        started = []
        engine.scan_started.connect(lambda idx: started.append(idx))

        self._run_engine(engine, config, hw)
        assert len(started) == 3
        assert started == [0, 1, 2]

    def test_signal_updated_emitted(self, qt_app):
        hw = make_mock_hw(n_pixels=16)
        config = make_config(n_delays=2, n_scans=1)
        engine = TransientAbsorptionEngine()

        updates = []
        engine.signal_updated.connect(lambda delay, wl, sig: updates.append(delay))

        self._run_engine(engine, config, hw)
        assert len(updates) == 2  # once per delay point


class TestEngineSaveSpectra:
    def _run_engine(self, engine, config, hw, writer=None, timeout=10.0):
        import time as _time
        from PySide6.QtWidgets import QApplication

        done = [False]
        engine.scan_completed.connect(lambda: done.__setitem__(0, True))
        engine.aborted.connect(lambda: done.__setitem__(0, True))
        engine.error.connect(lambda msg: done.__setitem__(0, True))
        engine.start_scan(config, hw, writer)
        start = _time.time()
        while not done[0] and _time.time() - start < timeout:
            QApplication.instance().processEvents()
            _time.sleep(0.005)

    def _make_config(self, n_delays=3, save_spectra_dir=None):
        return TAScanConfig(
            delay_list=[float(i) for i in range(n_delays)],
            n_averages=1,
            n_scans=1,
            acquisition_mode="boxcar",
            scan_direction="forward",
            sample_name="test",
            save_spectra_dir=save_spectra_dir,
        )

    def test_saves_spectrum_file_for_each_point(self, qt_app, tmp_path):
        hw = make_mock_hw(n_pixels=16)
        hw.get_wavelengths.return_value = np.linspace(400, 800, 16)
        config = self._make_config(n_delays=3, save_spectra_dir=str(tmp_path))
        engine = TransientAbsorptionEngine()
        self._run_engine(engine, config, hw)
        # Spectra are saved in a timestamped subfolder
        subdirs = [d for d in tmp_path.iterdir() if d.is_dir()]
        assert len(subdirs) == 1
        files = sorted(subdirs[0].glob("*.txt"))
        # 3 delays × 5 files each (delta, pump, ref, pump_std, ref_std)
        assert len(files) == 15

    def test_spectrum_file_is_two_column_text(self, qt_app, tmp_path):
        hw = make_mock_hw(n_pixels=4)
        hw.get_wavelengths.return_value = np.linspace(400, 800, 4)
        config = self._make_config(n_delays=1, save_spectra_dir=str(tmp_path))
        engine = TransientAbsorptionEngine()
        self._run_engine(engine, config, hw)
        subdirs = [d for d in tmp_path.iterdir() if d.is_dir()]
        assert len(subdirs) == 1
        # Pick the main delta spectrum file (scan000_pos*.txt)
        files = sorted(subdirs[0].glob("scan*.txt"))
        assert len(files) >= 1
        lines = files[0].read_text().splitlines()
        assert len(lines) == 4  # 4 pixels, no headers
        cols = lines[0].split("\t")
        assert len(cols) == 2

    def test_spectrum_files_have_no_comment_lines(self, qt_app, tmp_path):
        """All saved spectrum files must contain only data, no # comment lines."""
        hw = make_mock_hw(n_pixels=4)
        hw.get_wavelengths.return_value = np.linspace(400, 800, 4)
        config = self._make_config(n_delays=2, save_spectra_dir=str(tmp_path))
        engine = TransientAbsorptionEngine()
        self._run_engine(engine, config, hw)
        subdirs = [d for d in tmp_path.iterdir() if d.is_dir()]
        assert len(subdirs) == 1
        for f in subdirs[0].glob("*.txt"):
            for line in f.read_text().splitlines():
                assert not line.startswith("#"), (
                    f"Comment line found in {f.name}: {line!r}"
                )

    def test_no_files_when_save_dir_none(self, qt_app, tmp_path):
        hw = make_mock_hw(n_pixels=4)
        config = self._make_config(n_delays=2, save_spectra_dir=None)
        engine = TransientAbsorptionEngine()
        self._run_engine(engine, config, hw)
        assert list(tmp_path.iterdir()) == []


class TestEngineChopper2x2:
    """Engine correctly starts/stops trigger_gen and phase_reader."""

    def _run_engine(self, engine, config, hw, trigger_gen=None, phase_reader=None, timeout=10.0):
        import time as _time
        from PySide6.QtWidgets import QApplication
        done = [False]
        engine.scan_completed.connect(lambda: done.__setitem__(0, True))
        engine.aborted.connect(lambda: done.__setitem__(0, True))
        engine.error.connect(lambda msg: done.__setitem__(0, True))
        engine.start_scan(config, hw, writer=None, trigger_gen=trigger_gen, phase_reader=phase_reader)
        start = _time.time()
        while not done[0] and _time.time() - start < timeout:
            QApplication.instance().processEvents()
            _time.sleep(0.005)

    def _make_chopper_config(self):
        from andor_qt.ta.nidaq_phase import MockNIDAQChopper2x2Reader
        return TAScanConfig(
            delay_list=[0.0, 1.0],
            n_averages=1,
            n_scans=1,
            acquisition_mode="chopper_2x2",
            scan_direction="forward",
            sample_name="test",
        )

    def test_trigger_gen_started_and_stopped(self, qt_app):
        from unittest.mock import MagicMock
        from andor_qt.ta.nidaq_phase import MockNIDAQChopper2x2Reader
        hw = make_mock_hw()
        config = self._make_chopper_config()
        trigger_gen = MagicMock()
        reader = MockNIDAQChopper2x2Reader()
        self._run_engine(engine=TransientAbsorptionEngine(), config=config,
                         hw=hw, trigger_gen=trigger_gen, phase_reader=reader)
        trigger_gen.start.assert_called_once()
        trigger_gen.stop.assert_called_once()

    def test_phase_reader_started_and_stopped(self, qt_app):
        from unittest.mock import MagicMock
        from andor_qt.ta.nidaq_phase import MockNIDAQChopper2x2Reader
        hw = make_mock_hw()
        config = self._make_chopper_config()
        trigger_gen = MagicMock()
        reader = MagicMock()
        # Return matched-pair pattern for any requested length
        reader.read_tags.side_effect = lambda n: np.array(
            [1, 1, 0, 0] * ((n // 4) + 1), dtype=np.int8
        )[:n]
        self._run_engine(engine=TransientAbsorptionEngine(), config=config,
                         hw=hw, trigger_gen=trigger_gen, phase_reader=reader)
        reader.start.assert_called_once()
        reader.stop.assert_called_once()

    def test_scan_completes_with_chopper_2x2(self, qt_app):
        from andor_qt.ta.nidaq_phase import MockNIDAQChopper2x2Reader
        hw = make_mock_hw()
        config = self._make_chopper_config()
        trigger_gen = MagicMock()
        reader = MockNIDAQChopper2x2Reader()
        done = []
        engine = TransientAbsorptionEngine()
        engine.scan_completed.connect(lambda: done.append(True))
        self._run_engine(engine=engine, config=config, hw=hw,
                         trigger_gen=trigger_gen, phase_reader=reader)
        assert done


class TestEngineShotToShot:
    """Integration tests for shot_to_shot mode through the full engine."""

    def _run_engine(self, engine, config, hw, trigger_gen=None, phase_reader=None,
                    timeout=15.0):
        from PySide6.QtWidgets import QApplication
        import time as _time

        camera_settings = {"trigger_mode": "fast_external", "hbin": 1}
        engine.start_scan(config, hw, camera_settings=camera_settings,
                          trigger_gen=trigger_gen, phase_reader=phase_reader)
        done = [False]
        engine.scan_completed.connect(lambda: done.__setitem__(0, True))
        engine.error.connect(lambda e: done.__setitem__(0, True))
        start = _time.time()
        while not done[0] and _time.time() - start < timeout:
            QApplication.instance().processEvents()
            _time.sleep(0.005)

    def _make_s2s_config(self):
        return TAScanConfig(
            delay_list=[0.0, 1.0],
            n_averages=1,
            n_scans=1,
            acquisition_mode="shot_to_shot",
            scan_direction="forward",
            sample_name="test",
            crop_height=50,
        )

    def test_scan_completes_with_shot_to_shot(self, qt_app):
        from andor_qt.ta.nidaq_phase import MockNIDAQPhaseReader
        hw = make_mock_hw()
        # shot_to_shot uses crop mode RTA
        hw.camera.start_run_till_abort_crop.return_value = None
        hw.camera._current_hbin = 1
        config = self._make_s2s_config()
        reader = MockNIDAQPhaseReader()
        done = []
        engine = TransientAbsorptionEngine()
        engine.scan_completed.connect(lambda: done.append(True))
        self._run_engine(engine=engine, config=config, hw=hw,
                         phase_reader=reader)
        assert done

    def test_phase_reader_started_and_stopped(self, qt_app):
        from unittest.mock import MagicMock
        hw = make_mock_hw()
        hw.camera.start_run_till_abort_crop.return_value = None
        hw.camera._current_hbin = 1
        config = self._make_s2s_config()
        reader = MagicMock()
        reader.read_tags.side_effect = lambda n: np.array(
            [1, 0] * ((n // 2) + 1), dtype=np.int8
        )[:n]
        self._run_engine(engine=TransientAbsorptionEngine(), config=config,
                         hw=hw, phase_reader=reader)
        reader.start.assert_called_once()
        reader.stop.assert_called_once()

    def test_no_trigger_gen_needed(self, qt_app):
        from andor_qt.ta.nidaq_phase import MockNIDAQPhaseReader
        hw = make_mock_hw()
        hw.camera.start_run_till_abort_crop.return_value = None
        hw.camera._current_hbin = 1
        config = self._make_s2s_config()
        reader = MockNIDAQPhaseReader()
        done = []
        engine = TransientAbsorptionEngine()
        engine.scan_completed.connect(lambda: done.append(True))
        # No trigger_gen passed — shot_to_shot doesn't need one
        self._run_engine(engine=engine, config=config, hw=hw,
                         phase_reader=reader)
        assert done

    def test_signal_updated_emitted(self, qt_app):
        from andor_qt.ta.nidaq_phase import MockNIDAQPhaseReader
        hw = make_mock_hw()
        hw.camera.start_run_till_abort_crop.return_value = None
        hw.camera._current_hbin = 1
        config = self._make_s2s_config()
        reader = MockNIDAQPhaseReader()
        updates = []
        engine = TransientAbsorptionEngine()
        engine.signal_updated.connect(lambda d, w, s: updates.append(d))
        self._run_engine(engine=engine, config=config, hw=hw,
                         phase_reader=reader)
        assert len(updates) == 2  # 2 delay points


class TestTransientAbsorptionEnginePauseResume:
    def test_pause_resume_completes_scan(self, qt_app):
        from PySide6.QtWidgets import QApplication

        hw = make_mock_hw()
        config = make_config(n_delays=3, n_scans=1)
        engine = TransientAbsorptionEngine()

        done = [False]
        engine.scan_completed.connect(lambda: done.__setitem__(0, True))

        engine.start_scan(config, hw, None)
        time.sleep(0.01)
        engine.pause()
        time.sleep(0.05)
        engine.resume()

        start = time.time()
        while not done[0] and time.time() - start < 5.0:
            QApplication.instance().processEvents()
            time.sleep(0.005)

        assert done[0] is True


class TestStatusShowsPositionNotDelay:
    """Status messages should show stage position in µm, not delay in ps."""

    def _run_and_capture_status(self, qt_app, config=None, hw=None):
        """Run engine and capture all status_updated messages."""
        from PySide6.QtWidgets import QApplication

        if hw is None:
            hw = make_mock_hw()
        if config is None:
            config = make_config(n_delays=3, n_scans=1)

        engine = TransientAbsorptionEngine()
        status_msgs: List[str] = []
        engine.status_updated.connect(lambda msg: status_msgs.append(msg))

        done = [False]
        engine.scan_completed.connect(lambda: done.__setitem__(0, True))
        engine.aborted.connect(lambda: done.__setitem__(0, True))
        engine.error.connect(lambda _: done.__setitem__(0, True))

        engine.start_scan(config, hw)

        start = time.time()
        while not done[0] and time.time() - start < 10.0:
            QApplication.instance().processEvents()
            time.sleep(0.005)

        return status_msgs

    def test_status_contains_um_not_ps(self, qt_app):
        """Status messages with point info should show µm, not ps."""
        msgs = self._run_and_capture_status(qt_app)
        point_msgs = [m for m in msgs if m.startswith("pt ")]
        assert len(point_msgs) > 0, "No point status messages emitted"
        for msg in point_msgs:
            assert "µm" in msg, f"Status should show µm: {msg}"
            assert " ps " not in msg, f"Status should not show ps: {msg}"

    def test_progress_callback_shows_um(self, qt_app):
        """Progress callback during accumulation should show µm."""
        # Use chopper mode to trigger progress callbacks
        hw = make_mock_hw()
        from andor_qt.ta.nidaq_phase import MockNIDAQChopper2x2Reader
        config = TAScanConfig(
            delay_list=[0.0], n_averages=3, n_scans=1,
            acquisition_mode="chopper_2x2", scan_direction="forward",
            sample_name="test", shots_per_frame=2,
        )
        reader = MockNIDAQChopper2x2Reader()
        engine = TransientAbsorptionEngine()
        status_msgs: List[str] = []
        engine.status_updated.connect(lambda msg: status_msgs.append(msg))

        done = [False]
        engine.scan_completed.connect(lambda: done.__setitem__(0, True))
        engine.aborted.connect(lambda: done.__setitem__(0, True))
        engine.error.connect(lambda _: done.__setitem__(0, True))

        engine.start_scan(config, hw, phase_reader=reader)

        start = time.time()
        while not done[0] and time.time() - start < 10.0:
            from PySide6.QtWidgets import QApplication
            QApplication.instance().processEvents()
            time.sleep(0.005)

        point_msgs = [m for m in status_msgs if m.startswith("pt ")]
        for msg in point_msgs:
            assert "µm" in msg, f"Progress status should show µm: {msg}"

    def test_static_scan_status_shows_um(self, qt_app):
        """Static ON/OFF mode status messages should show µm."""
        hw = make_mock_hw()
        config = TAScanConfig(
            delay_list=[0.0, 1.0], n_averages=2, n_scans=1,
            acquisition_mode="static_onoff", scan_direction="forward",
            sample_name="test",
        )
        engine = TransientAbsorptionEngine()
        status_msgs: List[str] = []
        engine.status_updated.connect(lambda msg: status_msgs.append(msg))

        # Auto-respond to the "block pump" prompt
        engine.user_prompt.connect(lambda _: engine._worker._user_response.set())

        done = [False]
        engine.scan_completed.connect(lambda: done.__setitem__(0, True))
        engine.aborted.connect(lambda: done.__setitem__(0, True))
        engine.error.connect(lambda _: done.__setitem__(0, True))

        engine.start_scan(config, hw)

        start = time.time()
        while not done[0] and time.time() - start < 10.0:
            from PySide6.QtWidgets import QApplication
            QApplication.instance().processEvents()
            time.sleep(0.005)

        pass_msgs = [m for m in status_msgs if "Pass" in m and "pt " in m]
        for msg in pass_msgs:
            assert "µm" in msg, f"Static scan status should show µm: {msg}"
            assert " ps " not in msg, f"Static scan status should not show ps: {msg}"

    def test_monitor_status_shows_um(self, qt_app):
        """Monitor mode status should show position in µm."""
        from PySide6.QtWidgets import QApplication
        from andor_qt.ta.monitor_engine import TAMonitorEngine as MonitorEngine

        hw = make_mock_hw()
        config = TAScanConfig(
            delay_list=[0.0], n_averages=1,
            acquisition_mode="boxcar", scan_direction="forward",
            sample_name="test",
        )
        engine = MonitorEngine()
        status_msgs: List[str] = []
        engine.status_updated.connect(lambda msg: status_msgs.append(msg))

        engine.start_monitor(config, hw)

        # Let it run one cycle
        start = time.time()
        while time.time() - start < 3.0:
            QApplication.instance().processEvents()
            if any("Monitor cycle" in m for m in status_msgs):
                break
            time.sleep(0.05)

        engine.stop()
        start = time.time()
        while engine.is_running and time.time() - start < 3.0:
            QApplication.instance().processEvents()
            time.sleep(0.05)

        cycle_msgs = [m for m in status_msgs if "Monitor cycle" in m]
        for msg in cycle_msgs:
            assert "µm" in msg, f"Monitor status should show µm: {msg}"


class TestEngineStageAxisSelection:
    """Engine must apply config.stage_axis to the motion axis before scanning."""

    def _run_engine(self, engine, config, hw, timeout=10.0):
        import time as _time
        from PySide6.QtWidgets import QApplication

        done = [False]
        engine.scan_completed.connect(lambda: done.__setitem__(0, True))
        engine.aborted.connect(lambda: done.__setitem__(0, True))
        engine.error.connect(lambda msg: done.__setitem__(0, True))

        engine.start_scan(config, hw)

        start = _time.time()
        while not done[0] and _time.time() - start < timeout:
            QApplication.instance().processEvents()
            _time.sleep(0.005)

    def test_engine_calls_set_axis_hardware_index(self, qt_app):
        """Engine calls motion_manager.set_axis_hardware_index('delay', stage_axis)."""
        hw = make_mock_hw()
        config = make_config(n_delays=2, n_scans=1)
        config.stage_axis = 3  # non-default axis

        engine = TransientAbsorptionEngine()
        self._run_engine(engine, config, hw)

        hw.motion_manager.set_axis_hardware_index.assert_called_with("delay", 3)

    def test_engine_applies_stage_axis_1(self, qt_app):
        """Engine applies stage_axis=1 correctly."""
        hw = make_mock_hw()
        config = make_config(n_delays=2, n_scans=1)
        config.stage_axis = 1

        engine = TransientAbsorptionEngine()
        self._run_engine(engine, config, hw)

        hw.motion_manager.set_axis_hardware_index.assert_called_with("delay", 1)

    def test_engine_applies_stage_axis_for_static_scan(self, qt_app):
        """Engine applies stage_axis for static_onoff mode too."""
        hw = make_mock_hw()
        config = make_config(n_delays=2, n_scans=1)
        config.stage_axis = 3
        config.acquisition_mode = "static_onoff"

        engine = TransientAbsorptionEngine()

        # Static scan prompts user — abort after a brief delay to avoid hang
        done = [False]
        engine.user_prompt.connect(lambda _: engine.abort())
        engine.aborted.connect(lambda: done.__setitem__(0, True))
        engine.scan_completed.connect(lambda: done.__setitem__(0, True))
        engine.error.connect(lambda _: done.__setitem__(0, True))

        engine.start_scan(config, hw)

        import time as _time
        from PySide6.QtWidgets import QApplication
        start = _time.time()
        while not done[0] and _time.time() - start < 10.0:
            QApplication.instance().processEvents()
            _time.sleep(0.005)

        hw.motion_manager.set_axis_hardware_index.assert_called_with("delay", 3)
