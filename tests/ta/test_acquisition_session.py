"""Tests for AcquisitionSession context manager.

Verifies camera lifecycle, delegation to mode-specific acquisition,
and correct interaction with phase reader.
"""

from __future__ import annotations

from unittest.mock import MagicMock, call, patch

import numpy as np
import pytest

from andor_qt.ta.nidaq_phase import MockNIDAQChopper2x2Reader, MockNIDAQPhaseReader
from andor_qt.ta.scan_config import TAScanConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(mode: str = "chopper_2x2", n_averages: int = 3,
                 shots_per_frame: int = 2) -> TAScanConfig:
    return TAScanConfig(
        delay_list=[0.0],
        n_averages=n_averages,
        acquisition_mode=mode,
        scan_direction="forward",
        sample_name="test",
        shots_per_frame=shots_per_frame,
    )


def _make_hw_chopper(on_val, off_val, n_pairs, n_pixels=5):
    """Mock hw_manager for chopper_2x2 — alternating ON/OFF frames."""
    hw = MagicMock()
    hw.motion_manager = None
    frames = []
    for _ in range(n_pairs):
        frames.append(np.full(n_pixels, on_val, dtype=float))
        frames.append(np.full(n_pixels, off_val, dtype=float))
    frame_array = np.array(frames)
    hw.camera.start_run_till_abort.return_value = None
    hw.camera.get_buffered_frames.return_value = (frame_array, len(frames))
    hw.camera.abort_acquisition.return_value = None
    return hw


def _make_hw_software(shot_values):
    """Mock hw_manager for software/boxcar mode."""
    hw = MagicMock()
    hw.motion_manager = None
    it = iter(shot_values)
    hw.camera.get_spectrum.side_effect = lambda: np.array(next(it), dtype=float)
    return hw


# ---------------------------------------------------------------------------
# chopper_2x2 mode
# ---------------------------------------------------------------------------


class TestAcquisitionSessionChopper2x2:

    def test_enter_starts_camera_then_phase_reader_with_fire_trigger(self):
        from andor_qt.ta.acquisition import AcquisitionSession

        hw = _make_hw_chopper(1200, 1000, n_pairs=5)
        reader = MagicMock()
        reader.read_tags.return_value = np.array([1, 1, 0, 0] * 5, dtype=np.int8)
        config = _make_config("chopper_2x2")

        with AcquisitionSession(hw, config, phase_reader=reader, read_interval=0.01):
            # Camera starts FIRST (Fire output must be active)
            hw.camera.start_run_till_abort.assert_called_once()
            # Phase reader starts with Fire trigger from config
            reader.start.assert_called_once_with(
                start_trigger=config.nidaq_fire_trigger,
            )

    def test_exit_stops_camera(self):
        from andor_qt.ta.acquisition import AcquisitionSession

        hw = _make_hw_chopper(1200, 1000, n_pairs=5)
        reader = MockNIDAQChopper2x2Reader()
        config = _make_config("chopper_2x2")

        with AcquisitionSession(hw, config, phase_reader=reader, read_interval=0.01):
            pass
        hw.camera.abort_acquisition.assert_called_once()

    def test_exit_stops_camera_on_exception(self):
        from andor_qt.ta.acquisition import AcquisitionSession

        hw = _make_hw_chopper(1200, 1000, n_pairs=5)
        reader = MockNIDAQChopper2x2Reader()
        config = _make_config("chopper_2x2")

        with pytest.raises(ValueError):
            with AcquisitionSession(hw, config, phase_reader=reader, read_interval=0.01):
                raise ValueError("test error")
        hw.camera.abort_acquisition.assert_called_once()

    def test_camera_not_restarted_between_cycles(self):
        from andor_qt.ta.acquisition import AcquisitionSession

        hw = _make_hw_chopper(1200, 1000, n_pairs=10)
        reader = MockNIDAQChopper2x2Reader()
        config = _make_config("chopper_2x2", n_averages=3)

        with AcquisitionSession(hw, config, phase_reader=reader, read_interval=0.01) as session:
            session.acquire_one_cycle()
            session.acquire_one_cycle()
            session.acquire_one_cycle()

        # Camera started exactly once, not per cycle
        assert hw.camera.start_run_till_abort.call_count == 1

    def test_acquire_one_cycle_returns_delta_signal(self):
        from andor_qt.ta.acquisition import AcquisitionSession

        n = 5
        hw = _make_hw_chopper(1200, 1000, n_pairs=5)
        reader = MockNIDAQChopper2x2Reader()
        config = _make_config("chopper_2x2", n_averages=3)

        with AcquisitionSession(hw, config, phase_reader=reader, read_interval=0.01) as session:
            result = session.acquire_one_cycle()

        assert isinstance(result, np.ndarray)
        assert result.shape == (n,)
        # (1200 - 1000) / 1000 = 0.2
        assert result.mean() > 0

    def test_acquire_reads_tags(self):
        from andor_qt.ta.acquisition import AcquisitionSession

        hw = _make_hw_chopper(1200, 1000, n_pairs=5)
        reader = MagicMock()
        # Return alternating tags: [1,1,0,0,1,1,0,0,...] for spf=2
        n_frames = 10
        tags = np.array([1, 1, 0, 0] * (n_frames // 2), dtype=np.int8)[:n_frames * 2]
        reader.read_tags.return_value = tags
        reader.drain.return_value = None
        config = _make_config("chopper_2x2", n_averages=3, shots_per_frame=2)

        with AcquisitionSession(hw, config, phase_reader=reader, read_interval=0.01) as session:
            session.acquire_one_cycle()

        reader.read_tags.assert_called()
        # Should read n_chunk * spf tags
        call_args = reader.read_tags.call_args[0]
        assert call_args[0] == n_frames * 2  # n_chunk * spf

    def test_acquire_with_dark_subtraction(self):
        from andor_qt.ta.acquisition import AcquisitionSession

        n = 5
        hw = _make_hw_chopper(1200, 1000, n_pairs=5)
        reader = MockNIDAQChopper2x2Reader()
        config = _make_config("chopper_2x2", n_averages=3)
        dark = np.full(n, 100.0)

        with AcquisitionSession(hw, config, phase_reader=reader, read_interval=0.01) as session:
            result = session.acquire_one_cycle(dark=dark)

        assert isinstance(result, np.ndarray)
        # With dark: (1200-100 - (1000-100)) / (1000-100) = 200/900 ≈ 0.222
        assert abs(result.mean() - 200.0 / 900.0) < 0.05

    def test_acquire_with_raw_callback(self):
        from andor_qt.ta.acquisition import AcquisitionSession

        hw = _make_hw_chopper(1200, 1000, n_pairs=5)
        reader = MockNIDAQChopper2x2Reader()
        config = _make_config("chopper_2x2", n_averages=3)

        captured = {}

        def cb(pumped, ref, n_matched, n_discarded, n_frames):
            captured["pumped"] = pumped
            captured["ref"] = ref
            captured["n_matched"] = n_matched

        with AcquisitionSession(hw, config, phase_reader=reader, read_interval=0.01) as session:
            session.acquire_one_cycle(raw_callback=cb)

        assert "pumped" in captured
        assert captured["n_matched"] >= 1

    def test_applies_camera_settings_on_enter(self):
        from andor_qt.ta.acquisition import AcquisitionSession

        hw = _make_hw_chopper(1200, 1000, n_pairs=5)
        reader = MockNIDAQChopper2x2Reader()
        config = _make_config("chopper_2x2")
        settings = {"trigger_mode": "fast_external", "exposure_time": 0.0004}

        with AcquisitionSession(hw, config, camera_settings=settings,
                                phase_reader=reader, read_interval=0.01):
            pass

        hw.camera.apply_camera_settings.assert_called_once_with(settings)

    def test_zero_frames_raises(self):
        from andor_qt.ta.acquisition import AcquisitionSession

        hw = MagicMock()
        hw.motion_manager = None
        hw.camera.start_run_till_abort.return_value = None
        hw.camera.get_buffered_frames.return_value = (np.array([]), 0)
        hw.camera.abort_acquisition.return_value = None
        reader = MockNIDAQChopper2x2Reader()
        config = _make_config("chopper_2x2", n_averages=1)

        # Abort quickly so the incremental loop doesn't sleep 10s
        call_count = 0
        def _abort_after_2():
            nonlocal call_count
            call_count += 1
            return call_count > 2

        with AcquisitionSession(hw, config, phase_reader=reader, read_interval=0.01) as session:
            with pytest.raises(RuntimeError, match="aborted|no frames"):
                session.acquire_one_cycle(abort_check=_abort_after_2)


# ---------------------------------------------------------------------------
# software/boxcar mode
# ---------------------------------------------------------------------------


class TestAcquisitionSessionSoftware:

    def test_enter_does_not_start_rta(self):
        from andor_qt.ta.acquisition import AcquisitionSession

        shots = [[1200, 1000, 800], [1000, 900, 700]] * 3
        hw = _make_hw_software(shots)
        config = _make_config("boxcar", n_averages=3)

        with AcquisitionSession(hw, config):
            pass

        hw.camera.start_run_till_abort.assert_not_called()

    def test_exit_does_not_call_abort(self):
        from andor_qt.ta.acquisition import AcquisitionSession

        shots = [[1200, 1000, 800], [1000, 900, 700]] * 3
        hw = _make_hw_software(shots)
        config = _make_config("boxcar", n_averages=3)

        with AcquisitionSession(hw, config):
            pass

        hw.camera.abort_acquisition.assert_not_called()

    def test_acquire_one_cycle_returns_delta(self):
        from andor_qt.ta.acquisition import AcquisitionSession

        # 3 averages × 2 shots each = 6 spectra
        shots = [[1200, 1000, 800], [1000, 900, 700]] * 3
        hw = _make_hw_software(shots)
        config = _make_config("boxcar", n_averages=3)

        with AcquisitionSession(hw, config) as session:
            result = session.acquire_one_cycle()

        assert isinstance(result, np.ndarray)
        assert result.shape == (3,)

    def test_applies_camera_settings_on_enter(self):
        from andor_qt.ta.acquisition import AcquisitionSession

        shots = [[100], [90]] * 2
        hw = _make_hw_software(shots)
        config = _make_config("boxcar", n_averages=2)
        settings = {"trigger_mode": "internal"}

        with AcquisitionSession(hw, config, camera_settings=settings):
            pass

        hw.camera.apply_camera_settings.assert_called_once_with(settings)


# ---------------------------------------------------------------------------
# shot_to_shot mode
# ---------------------------------------------------------------------------


class TestAcquisitionSessionShotToShot:

    def test_shot_to_shot_delegates(self):
        from andor_qt.ta.acquisition import AcquisitionSession

        n = 4
        hw = MagicMock()
        hw.motion_manager = None
        # Build alternating ON/OFF frames
        frames = np.ones((10, n)) * 1000.0
        hw.camera.start_run_till_abort_crop.return_value = None
        hw.camera._current_hbin = 1
        hw.camera.get_buffered_frames.return_value = (frames, 10)
        hw.camera.abort_acquisition.return_value = None

        reader = MockNIDAQPhaseReader()
        config = _make_config("shot_to_shot", n_averages=3)

        with AcquisitionSession(hw, config, phase_reader=reader) as session:
            result = session.acquire_one_cycle()

        assert isinstance(result, np.ndarray)
        assert result.shape == (n,)

    def test_shot_to_shot_no_phase_reader_falls_back(self):
        """Without phase_reader, shot_to_shot should fall back to software mode."""
        from andor_qt.ta.acquisition import AcquisitionSession

        shots = [[1200, 1000], [1000, 900]] * 3
        hw = _make_hw_software(shots)
        config = _make_config("shot_to_shot", n_averages=3)

        with AcquisitionSession(hw, config, phase_reader=None) as session:
            result = session.acquire_one_cycle()

        assert isinstance(result, np.ndarray)


# ---------------------------------------------------------------------------
# Context manager protocol
# ---------------------------------------------------------------------------


class TestAcquisitionSessionProtocol:

    def test_context_manager_returns_self(self):
        from andor_qt.ta.acquisition import AcquisitionSession

        shots = [[100], [90]]
        hw = _make_hw_software(shots)
        config = _make_config("boxcar", n_averages=1)

        with AcquisitionSession(hw, config) as session:
            from andor_qt.ta.acquisition import AcquisitionSession as Cls
            assert isinstance(session, Cls)

    def test_no_phase_reader_for_chopper_falls_back_to_software(self):
        """chopper_2x2 without phase_reader should fall back to software mode."""
        from andor_qt.ta.acquisition import AcquisitionSession

        shots = [[1200, 1000], [1000, 900]] * 3
        hw = _make_hw_software(shots)
        config = _make_config("chopper_2x2", n_averages=3)

        with AcquisitionSession(hw, config, phase_reader=None) as session:
            result = session.acquire_one_cycle()

        assert isinstance(result, np.ndarray)
        # Should NOT have started RTA since there's no phase reader
        hw.camera.start_run_till_abort.assert_not_called()


# ---------------------------------------------------------------------------
# Incremental multi-chunk accumulation
# ---------------------------------------------------------------------------


def _make_multi_chunk_hw(on_val, off_val, frames_per_chunk, n_pixels=5,
                         initial_empty=0):
    """Mock hw_manager that returns frames in small batches.

    The first call is the drain call (always returns empty).
    Then ``initial_empty`` calls return (empty, 0) to simulate the camera
    not yet having frames ready.  After that, each call returns
    ``frames_per_chunk`` frames with alternating ON/OFF pattern.

    Args:
        on_val: Pixel value for pump-ON frames.
        off_val: Pixel value for pump-OFF frames.
        frames_per_chunk: Number of frames returned per non-empty call.
        n_pixels: Number of pixels per frame.
        initial_empty: Number of empty reads after the drain call.
    """
    hw = MagicMock()
    hw.motion_manager = None

    call_count = 0

    def _get_buffered(*args, **kwargs):
        nonlocal call_count
        call_count += 1

        # First call is the drain in __enter__ / top of _acquire_chopper_cycle
        if call_count <= 1:
            return (np.array([]), 0)

        # Simulate initial empty reads (camera hasn't accumulated frames yet)
        if call_count <= 1 + initial_empty:
            return (np.array([]), 0)

        # Return a batch of alternating ON/OFF frames
        frames = []
        for i in range(frames_per_chunk):
            val = on_val if i % 2 == 0 else off_val
            frames.append(np.full(n_pixels, val, dtype=float))
        return (np.array(frames), frames_per_chunk)

    hw.camera.start_run_till_abort.return_value = None
    hw.camera.get_buffered_frames.side_effect = _get_buffered
    hw.camera.abort_acquisition.return_value = None
    return hw


class TestIncrementalAccumulation:
    """Test the multi-chunk incremental accumulation path.

    The existing tests use mocks that return ALL frames at once, so the
    while loop in ``_acquire_chopper_cycle`` executes only one iteration.
    These tests verify the incremental path where frames arrive in small
    batches across multiple ``get_buffered_frames()`` calls.
    """

    def test_multi_chunk_accumulation_correct_delta(self):
        """Frames arrive in batches of 10; need 3+ reads to reach n_averages=20."""
        from andor_qt.ta.acquisition import AcquisitionSession

        on_val, off_val, n_pixels = 1200.0, 1000.0, 5
        frames_per_chunk = 10  # 5 ON + 5 OFF per chunk
        n_averages = 20        # need 20 ON and 20 OFF

        hw = _make_multi_chunk_hw(on_val, off_val, frames_per_chunk, n_pixels)
        reader = MockNIDAQChopper2x2Reader()
        config = _make_config("chopper_2x2", n_averages=n_averages)

        with AcquisitionSession(hw, config, phase_reader=reader, read_interval=0.01) as session:
            result = session.acquire_one_cycle()

        assert isinstance(result, np.ndarray)
        assert result.shape == (n_pixels,)
        # (1200 - 1000) / 1000 = 0.2
        np.testing.assert_allclose(result, 0.2, atol=1e-10)

        # Verify multiple chunks were read (drain + at least 3 data reads)
        assert hw.camera.get_buffered_frames.call_count >= 4

    def test_empty_reads_then_data(self):
        """Camera returns empty for first 2 data reads, then real frames."""
        from andor_qt.ta.acquisition import AcquisitionSession

        on_val, off_val, n_pixels = 1500.0, 1000.0, 4
        # 2 empty reads after drain, then 20 frames per chunk
        hw = _make_multi_chunk_hw(on_val, off_val, frames_per_chunk=20,
                                  n_pixels=n_pixels, initial_empty=2)
        reader = MockNIDAQChopper2x2Reader()
        config = _make_config("chopper_2x2", n_averages=5)

        with AcquisitionSession(hw, config, phase_reader=reader, read_interval=0.01) as session:
            result = session.acquire_one_cycle()

        assert isinstance(result, np.ndarray)
        assert result.shape == (n_pixels,)
        # (1500 - 1000) / 1000 = 0.5
        np.testing.assert_allclose(result, 0.5, atol=1e-10)

        # drain(1) + 2 empty + at least 1 data read = at least 4 calls
        assert hw.camera.get_buffered_frames.call_count >= 4

    def test_all_discarded_tags_raises(self):
        """All tags are mismatched (spf=2 with alternating [1,0]) -> RuntimeError."""
        from andor_qt.ta.acquisition import AcquisitionSession

        n_pixels = 5
        n_averages = 3
        frames_per_chunk = 40  # large chunk so min_check threshold is reached

        hw = MagicMock()
        hw.motion_manager = None

        call_count = 0

        def _get_buffered(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count <= 1:
                return (np.array([]), 0)
            frames = np.ones((frames_per_chunk, n_pixels)) * 1000.0
            return (frames, frames_per_chunk)

        hw.camera.start_run_till_abort.return_value = None
        hw.camera.get_buffered_frames.side_effect = _get_buffered
        hw.camera.abort_acquisition.return_value = None

        # Phase reader that returns alternating [1,0,1,0,...] — every tag
        # group will be [1,0] which is mismatched for spf=2
        reader = MagicMock()
        reader.drain.return_value = None

        def _mismatched_tags(n):
            return np.array([1, 0] * (n // 2 + 1), dtype=np.int8)[:n]

        reader.read_tags.side_effect = _mismatched_tags

        config = _make_config("chopper_2x2", n_averages=n_averages,
                              shots_per_frame=2)

        with AcquisitionSession(hw, config, phase_reader=reader, read_interval=0.01) as session:
            with pytest.raises(RuntimeError, match="no valid pairs"):
                session.acquire_one_cycle()

    def test_progress_callback_called_per_chunk(self):
        """progress_callback is called multiple times with increasing n_pairs."""
        from andor_qt.ta.acquisition import AcquisitionSession

        on_val, off_val, n_pixels = 1200.0, 1000.0, 5
        frames_per_chunk = 10
        n_averages = 20

        hw = _make_multi_chunk_hw(on_val, off_val, frames_per_chunk, n_pixels)
        reader = MockNIDAQChopper2x2Reader()
        config = _make_config("chopper_2x2", n_averages=n_averages)

        progress_calls = []

        def _on_progress(n_pairs, n_target, elapsed_s):
            progress_calls.append((n_pairs, n_target, elapsed_s))

        with AcquisitionSession(hw, config, phase_reader=reader, read_interval=0.01) as session:
            session.acquire_one_cycle(progress_callback=_on_progress)

        # Should have been called multiple times (once per chunk)
        assert len(progress_calls) >= 3

        # n_target should always be n_averages
        for _, n_target, _ in progress_calls:
            assert n_target == n_averages

        # n_pairs should be non-decreasing
        n_pairs_values = [c[0] for c in progress_calls]
        for i in range(1, len(n_pairs_values)):
            assert n_pairs_values[i] >= n_pairs_values[i - 1]

        # Final call should have n_pairs >= n_averages
        assert n_pairs_values[-1] >= n_averages

        # elapsed_s should be non-decreasing
        elapsed_values = [c[2] for c in progress_calls]
        for i in range(1, len(elapsed_values)):
            assert elapsed_values[i] >= elapsed_values[i - 1]

    def test_abort_check_during_accumulation(self):
        """abort_check triggers after 2 data reads -> RuntimeError("aborted")."""
        from andor_qt.ta.acquisition import AcquisitionSession

        on_val, off_val, n_pixels = 1200.0, 1000.0, 5
        frames_per_chunk = 4  # small chunks, need many reads
        n_averages = 100       # high target so we never finish naturally

        hw = _make_multi_chunk_hw(on_val, off_val, frames_per_chunk, n_pixels)
        reader = MockNIDAQChopper2x2Reader()
        config = _make_config("chopper_2x2", n_averages=n_averages)

        data_reads = 0

        def _abort_after_2_data_reads():
            # Count how many times get_buffered_frames has been called
            # with actual data returned (call_count > 1 in the side_effect)
            nonlocal data_reads
            # Each abort_check call is at top of while loop, before the read.
            # We count iterations to determine when to abort.
            data_reads += 1
            return data_reads > 2

        with AcquisitionSession(hw, config, phase_reader=reader, read_interval=0.01) as session:
            with pytest.raises(RuntimeError, match="aborted"):
                session.acquire_one_cycle(abort_check=_abort_after_2_data_reads)


class TestChopper2x2Heartbeat:
    """Heartbeat logs let us spot stalls inside the SDK that the per-iteration
    safety nets miss (e.g. GetImages16 hangs without returning). Each cycle
    logs at the start and then again every ``_heartbeat_interval`` seconds
    while the read loop spins."""

    def test_logs_cycle_start(self, caplog):
        """A 'cycle start' marker is logged at the top of every cycle."""
        from andor_qt.ta.acquisition import AcquisitionSession
        hw = _make_multi_chunk_hw(1200.0, 1000.0, frames_per_chunk=10)
        reader = MockNIDAQChopper2x2Reader()
        config = _make_config("chopper_2x2", n_averages=5)

        with caplog.at_level("INFO", logger="andor_qt.ta.acquisition"):
            with AcquisitionSession(hw, config, phase_reader=reader,
                                    read_interval=0.001) as session:
                session.acquire_one_cycle()

        markers = [r for r in caplog.records if "cycle start" in r.getMessage()]
        assert len(markers) >= 1, (
            "Expected a 'cycle start' log; got: "
            + ", ".join(r.getMessage()[:80] for r in caplog.records)
        )

    def test_heartbeat_includes_state(self, caplog):
        """Heartbeat message names frames/ON/OFF/empty so a stall is diagnosable."""
        from andor_qt.ta.acquisition import AcquisitionSession
        hw = _make_multi_chunk_hw(1200.0, 1000.0, frames_per_chunk=4,
                                  initial_empty=3)
        reader = MockNIDAQChopper2x2Reader()
        config = _make_config("chopper_2x2", n_averages=20)

        with caplog.at_level("INFO", logger="andor_qt.ta.acquisition"):
            with AcquisitionSession(hw, config, phase_reader=reader,
                                    read_interval=0.001) as session:
                session._heartbeat_interval = 0.0  # log every iteration
                session.acquire_one_cycle()

        hb = [r.getMessage() for r in caplog.records
              if "heartbeat" in r.getMessage()]
        assert len(hb) >= 1, "Expected at least one heartbeat log"
        # First heartbeat should reference the diagnostic fields
        msg = hb[0]
        assert "frames" in msg
        assert "ON" in msg
        assert "OFF" in msg
        assert "empty" in msg

    def test_heartbeat_throttled_by_interval(self, caplog):
        """With a large interval, no heartbeat fires for a short fast acquisition."""
        from andor_qt.ta.acquisition import AcquisitionSession
        hw = _make_multi_chunk_hw(1200.0, 1000.0, frames_per_chunk=100)
        reader = MockNIDAQChopper2x2Reader()
        config = _make_config("chopper_2x2", n_averages=10)

        with caplog.at_level("INFO", logger="andor_qt.ta.acquisition"):
            with AcquisitionSession(hw, config, phase_reader=reader,
                                    read_interval=0.001) as session:
                session._heartbeat_interval = 100.0  # very long
                session.acquire_one_cycle()

        hb = [r for r in caplog.records if "heartbeat" in r.getMessage()]
        assert len(hb) == 0, (
            f"Expected no heartbeats with interval=100s, got {len(hb)}: "
            + "; ".join(r.getMessage()[:80] for r in hb)
        )
