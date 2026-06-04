"""Low-level acquisition helper for TA measurements.

``acquire_delta_signal_at_delay`` moves the delay stage to a given position
and acquires a ΔI/I₀ spectrum by averaging pump-on/pump-off pairs.

Hardware chopper mode (recommended)
------------------------------------
Pass a ``phase_reader`` (``NIDAQPhaseReader`` or ``MockNIDAQPhaseReader``).
For each shot pair the phase reader supplies a tag from the NI DAQ digital
input line (1 = pump-on, 0 = pump-off), so pump-on/off assignment is
determined by the chopper hardware rather than by shot ordering.

Software fallback (``phase_reader=None``)
-----------------------------------------
Two sequential spectra are acquired per pair; the first is treated as
pump-on and the second as pump-off.  This is kept for backward compatibility
but is less robust than hardware tagging.

This function is used by both ``TransientAbsorptionEngine`` (scan loop) and
``T0Finder`` (t0 search).
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Callable, Optional

import numpy as np

from andor_qt.ta.delta_signal import average_delta_signal, background_subtract, compute_delta_signal
from andor_qt.ta.scan_config import TAScanConfig

log = logging.getLogger(__name__)

# Last acquisition statistics — populated after each acquisition call.
# Keys: pump_mean, pump_std, ref_mean, ref_std, delta_std, n_on, n_off
last_acquisition_stats: dict = {}


def acquire_delta_signal_at_delay(
    delay_ps: float,
    hw_manager: Any,
    config: TAScanConfig,
    dark: Optional[np.ndarray] = None,
    camera_settings: Optional[dict[str, Any]] = None,
    phase_reader: Any = None,
    raw_callback: Optional[Callable[[np.ndarray, np.ndarray, int, int, int], None]] = None,
) -> np.ndarray:
    """Acquire ΔI/I₀ spectrum at a specific time delay.

    Moves the delay stage to ``delay_ps``, then acquires ``n_averages``
    pump-on/pump-off spectrum pairs and returns the averaged ΔI/I₀.

    Args:
        delay_ps: Target delay in picoseconds.
        hw_manager: Hardware manager with ``.camera``, ``.motion`` attributes.
        config: Scan configuration (``n_averages``, ``acquisition_mode``).
        dark: Optional dark spectrum to subtract before computing ΔI/I₀.
        camera_settings: Optional dict passed to camera.apply_camera_settings()
            before acquisition. If None, current camera settings are unchanged.
        phase_reader: Optional NIDAQPhaseReader (or mock).  When provided,
            each shot is tagged by reading one sample from the NI DAQ digital
            input, and ``ChopperSync`` assigns pump-on/off accordingly.
            When ``None``, the first spectrum of each pair is taken as pump-on
            and the second as pump-off (software fallback).

    Returns:
        Averaged ΔI/I₀ spectrum (1-D numpy array).
    """
    # Clear stale stats so they don't leak from a previous call on error
    last_acquisition_stats.clear()

    # Apply user-selected stage axis, then move to target delay
    mm = getattr(hw_manager, "motion_manager", None)
    if mm is not None and hasattr(mm, "set_axis_hardware_index"):
        mm.set_axis_hardware_index("delay", config.stage_axis)
    axis = mm.get_axis("delay") if mm is not None else None
    if axis is not None:
        axis.position_ps = delay_ps

    # Apply camera settings once before the averaging loop
    if camera_settings is not None:
        apply = getattr(hw_manager.camera, "apply_camera_settings", None)
        if callable(apply):
            apply(camera_settings)

    if config.acquisition_mode == "shot_to_shot" and phase_reader is not None:
        return _acquire_shot_to_shot(hw_manager, config, dark, phase_reader, raw_callback=raw_callback)
    if config.acquisition_mode == "chopper_2x2" and phase_reader is not None:
        # Single-use session: camera starts, acquires one cycle, stops.
        # For multi-point scans, use AcquisitionSession directly to avoid
        # restarting the camera per point (which breaks phase stability).
        with AcquisitionSession(hw_manager, config, camera_settings=camera_settings,
                                phase_reader=phase_reader) as session:
            return session.acquire_one_cycle(dark=dark, raw_callback=raw_callback)
    return _acquire_software(hw_manager, config, dark, raw_callback=raw_callback)


# ---------------------------------------------------------------------------
# Chopper frame processing (used by AcquisitionSession and monitor)
# ---------------------------------------------------------------------------

def _process_chopper_frames(
    frames: np.ndarray,
    tags: np.ndarray,
    config: TAScanConfig,
    dark: Optional[np.ndarray] = None,
    raw_callback: Optional[Callable] = None,
) -> np.ndarray:
    """Process pre-read frames and tags into a delta-I/I0 spectrum.

    This is the core computation shared by both the chopper_2x2 acquisition
    (which starts/stops the camera) and the continuous monitor mode (which
    keeps the camera running).

    Args:
        frames: 2-D array of shape (n_frames, n_pixels).
        tags: 1-D array of phase tags (one per laser shot).
        config: Scan config with shots_per_frame, n_averages.
        dark: Optional dark spectrum to subtract.
        raw_callback: Optional callback for raw pump/ref display.

    Returns:
        Averaged delta-I/I0 spectrum (1-D array).
    """
    spf = getattr(config, "shots_per_frame", 2)
    n_frames = len(frames)

    # Group tags by shots_per_frame and detect alignment offset
    best_offset = 0
    best_matched = -1
    for offset in range(spf):
        usable = (len(tags) - offset) // spf
        if usable < 1:
            continue
        grp = tags[offset:offset + usable * spf].reshape(usable, spf)
        n_matched = int((grp == grp[:, :1]).all(axis=1).sum())
        if n_matched > best_matched:
            best_matched = n_matched
            best_offset = offset

    usable = min(n_frames, (len(tags) - best_offset) // spf)
    tag_groups = tags[best_offset:best_offset + usable * spf].reshape(usable, spf)
    frames = frames[:usable]
    n_read = len(frames)

    # Separate matched frames by pump state
    matched_mask = (tag_groups == tag_groups[:, :1]).all(axis=1)
    n_discarded = int(n_read - matched_mask.sum())
    matched_frames = frames[matched_mask]
    matched_tags = tag_groups[matched_mask, 0]

    # Tag convention: P0.0=1 → pump-ON, P0.0=0 → pump-OFF.
    # If config.swap_tags is True, reverse the assignment — useful when the
    # chopper REF OUT polarity is inverted relative to the beam path.
    swap = bool(getattr(config, "swap_tags", False))
    on_tag, off_tag = (0, 1) if swap else (1, 0)
    on_frames = matched_frames[matched_tags == on_tag]
    off_frames = matched_frames[matched_tags == off_tag]
    n_pairs = min(len(on_frames), len(off_frames), config.n_averages)

    log.info(
        f"chopper_2x2 tag stats: {n_read} frames, {n_discarded} discarded, "
        f"{len(on_frames)} ON, {len(off_frames)} OFF, {n_pairs} pairs  |  "
        f"ON mean={on_frames.mean():.1f}, OFF mean={off_frames.mean():.1f}"
        if len(on_frames) > 0 and len(off_frames) > 0 else
        f"chopper_2x2 tag stats: {n_read} frames, {n_discarded} discarded, "
        f"{len(on_frames)} ON, {len(off_frames)} OFF"
    )

    if n_pairs == 0:
        raise RuntimeError(
            f"chopper_2x2: {n_read} frames, {n_discarded} discarded, "
            f"0 valid pairs -- check chopper phase sync"
        )

    pumped = on_frames[:n_pairs]
    ref = off_frames[:n_pairs]

    if dark is not None:
        pumped = pumped - dark[np.newaxis, :]
        ref = ref - dark[np.newaxis, :]

    ref_safe = np.where(ref == 0, 1.0, ref)
    delta = (pumped - ref) / ref_safe
    mean = delta.mean(axis=0)

    last_acquisition_stats.update({
        "pump_mean": on_frames.mean(axis=0),
        "pump_std": on_frames.std(axis=0),
        "ref_mean": off_frames.mean(axis=0),
        "ref_std": off_frames.std(axis=0),
        "delta_std": delta.std(axis=0),
        "n_on": len(on_frames),
        "n_off": len(off_frames),
    })

    if raw_callback is not None:
        raw_callback(on_frames.mean(axis=0), off_frames.mean(axis=0),
                     n_pairs, n_discarded, n_read)

    return mean


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _acquire_shot_to_shot(
    hw_manager: Any,
    config: TAScanConfig,
    dark: Optional[np.ndarray],
    phase_reader: Any,
    raw_callback: Optional[Callable[[np.ndarray, np.ndarray, int, int, int], None]] = None,
) -> np.ndarray:
    """Acquire using shot-to-shot mode — 1 kHz single-shot crop-mode acquisition.

    The camera runs in isolated crop mode (reduced rows for <1 ms readout)
    with a 1 kHz external trigger (one frame per laser shot).  Each frame
    gets a single P0.0 tag: 1 = pump-ON, 0 = pump-OFF.  Frames are sorted
    by pump state and paired for ΔI/I₀ computation.

    For large n_averages, acquisition is split into chunks to avoid
    overflowing the camera circular buffer.
    """
    camera = hw_manager.camera
    n_target = int(config.n_averages * 2.2) + 10
    # Max frames per chunk — stay well below circular buffer limit (~15000 frames)
    buf_size = getattr(camera, "get_circular_buffer_size", lambda: 12000)()
    max_chunk = max(1000, int(buf_size * 0.8))
    log.info(f"shot_to_shot: n_target={n_target}, buf_size={buf_size}, max_chunk={max_chunk}")

    hbin = getattr(camera, "_current_hbin", 1)
    all_frames = []
    all_tags = []
    remaining = n_target
    chunk_idx = 0

    while remaining > 0:
        chunk = min(remaining, max_chunk)

        camera.start_run_till_abort_crop(
            crop_height=config.crop_height,
            hbin=hbin,
        )
        phase_reader.drain()

        try:
            # Wait for frames: 1 ms/frame at 1 kHz (with overlap mode) + 20% margin
            wait_s = (chunk * 1.0) / 1000.0 * 1.2 + 0.05
            time.sleep(wait_s)
            frames, n_read = camera.get_buffered_frames()
        finally:
            camera.abort_acquisition()

        log.info(f"  chunk {chunk_idx}: requested={chunk}, got={n_read}, wait={wait_s:.2f}s, remaining={remaining}")
        chunk_idx += 1

        if n_read == 0:
            break

        # Read available tags — may be slightly more or fewer than n_read
        read_avail = getattr(phase_reader, "read_available_tags", None)
        tags = read_avail() if callable(read_avail) else np.array([], dtype=np.int8)
        if len(tags) == 0:
            # Fallback: blocking read for exactly n_read tags
            tags = phase_reader.read_tags(n_read)
        if len(tags) == 0:
            # No tags available from any source — abort to prevent infinite loop
            break

        # Align frame and tag counts
        n_use = min(n_read, len(tags))
        if n_use > 0:
            all_frames.append(frames[:n_use])
            all_tags.append(tags[:n_use])
        remaining -= n_use

    if not all_frames:
        raise RuntimeError("shot_to_shot: no frames acquired — check trigger")

    frames = np.concatenate(all_frames)
    tags = np.concatenate(all_tags)
    # Align: use the shorter of frames and tags
    n_read = min(len(frames), len(tags))
    frames = frames[:n_read]
    tags = tags[:n_read]

    if n_read == 0:
        raise RuntimeError("shot_to_shot: no frames acquired — check trigger")

    # Separate by pump state
    on_mask = tags == 1
    off_mask = tags == 0
    on_frames = frames[on_mask]
    off_frames = frames[off_mask]
    n_pairs = min(len(on_frames), len(off_frames), config.n_averages)

    if n_pairs == 0:
        raise RuntimeError(
            f"shot_to_shot: {n_read} frames, "
            f"{int(on_mask.sum())} ON, {int(off_mask.sum())} OFF — "
            f"0 valid pairs"
        )

    pumped = on_frames[:n_pairs]
    ref = off_frames[:n_pairs]

    if dark is not None:
        pumped = pumped - dark[np.newaxis, :]
        ref = ref - dark[np.newaxis, :]

    ref_safe = np.where(ref == 0, 1.0, ref)
    delta = (pumped - ref) / ref_safe
    mean = delta.mean(axis=0)

    last_acquisition_stats.update({
        "pump_mean": on_frames.mean(axis=0),
        "pump_std": on_frames.std(axis=0),
        "ref_mean": off_frames.mean(axis=0),
        "ref_std": off_frames.std(axis=0),
        "delta_std": delta.std(axis=0),
        "n_on": len(on_frames),
        "n_off": len(off_frames),
    })

    if raw_callback is not None:
        raw_callback(on_frames.mean(axis=0), off_frames.mean(axis=0),
                     n_pairs, 0, 2 * n_pairs)

    log.info(
        f"shot_to_shot: {n_pairs} pairs from {n_read} frames "
        f"({int(on_mask.sum())} ON, {int(off_mask.sum())} OFF, "
        f"{wait_s:.2f}s accumulation)"
    )
    return mean


def _acquire_software(
    hw_manager: Any,
    config: TAScanConfig,
    dark: Optional[np.ndarray],
    raw_callback: Optional[Callable[[np.ndarray, np.ndarray, int, int, int], None]] = None,
) -> np.ndarray:
    """Acquire using software alternation (first shot = pump-on)."""
    delta_signal_list = []
    pump_sum = None
    ref_sum = None

    for i in range(config.n_averages):
        pumped = np.asarray(hw_manager.camera.get_spectrum(), dtype=float)
        ref = np.asarray(hw_manager.camera.get_spectrum(), dtype=float)

        if dark is not None:
            pumped = background_subtract(pumped, dark)
            ref = background_subtract(ref, dark)

        # Accumulate running averages for raw callback and stats
        if pump_sum is None:
            pump_sum = pumped.copy()
            ref_sum = ref.copy()
        else:
            pump_sum += pumped
            ref_sum += ref

        delta_signal_list.append(compute_delta_signal(pumped, ref))

    n = config.n_averages
    pump_avg = pump_sum / n if pump_sum is not None else np.array([])
    ref_avg = ref_sum / n if ref_sum is not None else np.array([])

    # Populate stats with per-pixel arrays (consistent with hardware modes)
    mean, _ = average_delta_signal(delta_signal_list)
    delta_std = np.std(np.stack(delta_signal_list), axis=0) if len(delta_signal_list) > 1 else np.zeros_like(mean)
    last_acquisition_stats.update({
        "pump_mean": pump_avg,
        "pump_std": pump_avg,  # single-pair std not tracked; use pump_avg as placeholder
        "ref_mean": ref_avg,
        "ref_std": ref_avg,
        "delta_std": delta_std,
        "n_on": n,
        "n_off": n,
    })

    if raw_callback is not None and len(pump_avg) > 0:
        raw_callback(pump_avg, ref_avg, n, 0, 2 * n)

    return mean


# ---------------------------------------------------------------------------
# Long-average utility (shared by scan and monitor engines)
# ---------------------------------------------------------------------------

def acquire_long_average(
    camera: Any,
    n_target: int,
    abort_event: threading.Event,
    progress_cb: Optional[Callable[[np.ndarray, int, int], None]] = None,
    frame_period_s: float = 0.002,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Acquire many frames and return running-mean statistics.

    Uses chunked ``start_run_till_abort`` + ``get_buffered_frames`` with
    incremental sum / sum-of-squares for memory-efficient statistics.

    Args:
        camera: Camera object with ``start_run_till_abort()``,
            ``get_buffered_frames()``, ``abort_acquisition()``, and
            ``get_circular_buffer_size()`` methods.
        n_target: Number of frames to accumulate.
        abort_event: Set this event to abort early.
        progress_cb: Optional ``(running_mean, collected, n_target)`` callback
            invoked after each chunk for live progress updates.
        frame_period_s: Expected time per frame in seconds. Used to calculate
            how long to wait for a chunk of frames. Default 2 ms (500 Hz).
            For external trigger at 1 kHz, pass 0.001.

    Returns:
        ``(mean, std, count)`` where ``mean`` and ``std`` are 1-D arrays.

    Raises:
        RuntimeError: If no frames were acquired.
    """
    buf_size = getattr(camera, "get_circular_buffer_size", lambda: 12000)()
    max_chunk = max(1000, int(buf_size * 0.8))

    running_sum = None
    running_sum_sq = None
    collected = 0

    while collected < n_target and not abort_event.is_set():
        chunk = min(n_target - collected, max_chunk)

        camera.start_run_till_abort()
        try:
            wait_s = chunk * frame_period_s * 1.2 + 0.05
            time.sleep(wait_s)
            frames, n_read = camera.get_buffered_frames()
        finally:
            camera.abort_acquisition()

        if n_read == 0:
            break

        chunk_sum = frames.sum(axis=0)
        chunk_sum_sq = (frames.astype(np.float64) ** 2).sum(axis=0)
        if running_sum is None:
            running_sum = chunk_sum
            running_sum_sq = chunk_sum_sq
        else:
            running_sum += chunk_sum
            running_sum_sq += chunk_sum_sq
        collected += n_read

        if progress_cb is not None and collected > 0:
            progress_cb(running_sum / collected, collected, n_target)

    if running_sum is None or collected == 0:
        raise RuntimeError("Long average: no frames acquired")

    mean = running_sum / collected
    variance = running_sum_sq / collected - mean ** 2
    std = np.sqrt(np.maximum(variance, 0.0))
    return mean, std, collected


# ---------------------------------------------------------------------------
# Frame period estimation
# ---------------------------------------------------------------------------

def _compute_frame_period_s(camera_settings: Optional[dict] = None) -> float:
    """Estimate the time per frame from camera settings.

    For external trigger modes, the frame period is determined by the trigger
    source (e.g. 1 kHz laser sync → 1 ms, 500 Hz chopper → 2 ms), not by
    the camera's internal exposure + readout. For internal trigger, uses
    exposure_time + readout_time.

    Returns:
        Seconds per frame.
    """
    if camera_settings is None:
        return 0.002

    trigger_mode = camera_settings.get("trigger_mode", "internal")

    # External trigger: frame rate set by trigger source, not camera internals
    if trigger_mode == "fast_external":
        # Fast external with overlap: limited by trigger rate (typically 1 kHz)
        return 0.001  # 1 ms at 1 kHz laser sync
    elif trigger_mode == "external":
        # Non-fast external: exposure + readout (no overlap)
        pass  # fall through to calculation below

    exposure_s = camera_settings.get("exposure_time", 0.002)

    try:
        from andor_qt.utils.readout_time import calculate_readout_time_ms
        vs_idx = camera_settings.get("vs_speed", 1)
        hs_idx = camera_settings.get("hs_speed", 1)
        hbin = camera_settings.get("hbin", 1)
        if isinstance(hbin, str):
            hbin = int(hbin.replace("x", ""))
        readout_s = calculate_readout_time_ms("fvb", 200, 1600, vs_idx, hs_idx, hbin) / 1000.0
    except Exception:
        readout_s = 0.001

    return exposure_s + readout_s


# ---------------------------------------------------------------------------
# Static (bulk) acquisition — shared by scan and monitor static paths
# ---------------------------------------------------------------------------

def acquire_static_at_delay(
    hw_manager: Any,
    n_frames: int,
    abort_event: "threading.Event",
    dark: Optional[np.ndarray] = None,
    camera_settings: Optional[dict[str, Any]] = None,
    progress_cb: Optional[Callable[[np.ndarray, int, int], None]] = None,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Acquire bulk frames at the current position for static ON/OFF mode.

    Handles camera settings, dark subtraction, and stats population.
    The caller is responsible for moving the stage before calling this.

    Args:
        hw_manager: Hardware manager with ``.camera`` attribute.
        n_frames: Number of frames to average.
        abort_event: Set to abort early.
        dark: Optional dark spectrum to subtract from the mean.
        camera_settings: Optional dict passed to ``apply_camera_settings()``.
        progress_cb: Optional ``(running_mean, collected, n_target)`` callback.

    Returns:
        ``(mean, std, count)`` — dark-subtracted mean, std, and frame count.
    """
    # Apply camera settings
    if camera_settings is not None:
        apply = getattr(hw_manager.camera, "apply_camera_settings", None)
        if callable(apply):
            apply(camera_settings)

    # Compute frame period from camera settings for accurate wait time
    frame_period_s = _compute_frame_period_s(camera_settings)

    # Acquire
    mean, std, count = acquire_long_average(
        hw_manager.camera, n_frames, abort_event,
        progress_cb=progress_cb, frame_period_s=frame_period_s,
    )

    # Dark subtraction
    if dark is not None:
        mean = background_subtract(mean, dark)

    # Populate stats with per-pixel arrays (consistent with hardware modes)
    last_acquisition_stats.update({
        "pump_mean": mean,
        "pump_std": std,
        "ref_mean": np.zeros_like(mean),
        "ref_std": np.zeros_like(std),
        "n_on": count,
        "n_off": 0,
    })

    return mean, std, count


# ---------------------------------------------------------------------------
# AcquisitionSession — unified camera lifecycle for scan and monitor
# ---------------------------------------------------------------------------


class AcquisitionSession:
    """Context manager owning the camera lifecycle across multiple acquisitions.

    For ``chopper_2x2``: camera starts once in ``__enter__``, reads buffered
    frames per ``acquire_one_cycle()``, stops in ``__exit__``.  This
    preserves the phase relationship between camera frames and chopper tags.

    For ``shot_to_shot``: delegates to ``_acquire_shot_to_shot`` per cycle
    (which manages its own crop-mode camera start/stop).

    For ``boxcar``/software: delegates to ``_acquire_software`` per cycle
    (which uses ``get_spectrum()`` per shot, no RTA).

    Usage::

        with AcquisitionSession(hw, config, camera_settings, phase_reader) as s:
            for delay_ps in delays:
                axis.position_ps = delay_ps
                delta = s.acquire_one_cycle(dark=dark, raw_callback=cb)

    Args:
        hw_manager: Hardware manager with ``.camera`` attribute.
        config: Scan configuration (``acquisition_mode``, ``n_averages``, etc.).
        camera_settings: Optional dict passed to ``apply_camera_settings()``.
        phase_reader: Optional phase reader for hardware-tagged modes.
    """

    def __init__(
        self,
        hw_manager: Any,
        config: TAScanConfig,
        camera_settings: Optional[dict[str, Any]] = None,
        phase_reader: Any = None,
        read_interval: float = 0.5,
    ) -> None:
        self._hw = hw_manager
        self._config = config
        self._camera_settings = camera_settings
        self._phase_reader = phase_reader
        self._camera_running = False
        self._read_interval = read_interval
        # Wall-clock interval between diagnostic heartbeat logs emitted by
        # ``_acquire_chopper_cycle``.  Set to 0 to log every iteration; large
        # to disable.  Lets us tell whether a long acquisition is iterating
        # normally or blocked inside the SDK without returning.
        self._heartbeat_interval = 1.0
        self._is_chopper = (
            config.acquisition_mode == "chopper_2x2"
            and phase_reader is not None
        )

    def __enter__(self) -> "AcquisitionSession":
        # Apply camera settings once
        if self._camera_settings is not None:
            apply = getattr(self._hw.camera, "apply_camera_settings", None)
            if callable(apply):
                apply(self._camera_settings)

        if self._is_chopper:
            # Phase reader arms FIRST with Camera Fire (PFI13) as start
            # trigger — it waits for the rising edge before acquiring.
            # Camera starts AFTER so the first Fire pulse is guaranteed
            # to be caught by the already-armed reader.
            fire = getattr(self._config, "nidaq_fire_trigger", None)
            self._phase_reader.start(start_trigger=fire)
            self._hw.camera.start_run_till_abort()
            self._camera_running = True

        return self

    def __exit__(self, *exc: object) -> None:
        if self._camera_running:
            self._hw.camera.abort_acquisition()
            self._camera_running = False

    def acquire_one_cycle(
        self,
        dark: Optional[np.ndarray] = None,
        raw_callback: Optional[Callable] = None,
        progress_callback: Optional[Callable[[int, int, float], None]] = None,
        abort_check: Optional[Callable[[], bool]] = None,
    ) -> np.ndarray:
        """Acquire one delta-signal measurement at the current position.

        For ``chopper_2x2``, reads frames from the already-running camera.
        For other modes, delegates to the existing mode-specific functions.

        Args:
            dark: Optional dark spectrum to subtract.
            raw_callback: Optional ``(pumped, ref, n_matched, n_discarded,
                n_frames)`` callback for live display.
            progress_callback: Optional ``(n_accumulated, n_target, elapsed_s)``
                callback for progress updates during frame accumulation.
            abort_check: Optional callable returning True if acquisition
                should be aborted early.

        Returns:
            Averaged ΔI/I₀ spectrum (1-D numpy array).
        """
        config = self._config

        if self._is_chopper:
            return self._acquire_chopper_cycle(
                dark, raw_callback, progress_callback, abort_check,
            )
        if config.acquisition_mode == "shot_to_shot" and self._phase_reader is not None:
            return _acquire_shot_to_shot(
                self._hw, config, dark, self._phase_reader,
                raw_callback=raw_callback,
            )
        return _acquire_software(self._hw, config, dark, raw_callback=raw_callback)

    def _acquire_chopper_cycle(
        self,
        dark: Optional[np.ndarray],
        raw_callback: Optional[Callable],
        progress_callback: Optional[Callable[[int, int, float], None]] = None,
        abort_check: Optional[Callable[[], bool]] = None,
    ) -> np.ndarray:
        """Read one cycle of frames using incremental accumulation.

        Instead of sleeping for the full acquisition time and reading all
        frames at once (which overflows the circular buffer for large
        n_averages), this method reads frames in ~1-second chunks and
        maintains running ON/OFF sums.

        Reports progress via ``progress_callback(n_pairs, n_target, elapsed_s)``
        every second.
        """
        camera = self._hw.camera
        config = self._config
        spf = getattr(config, "shots_per_frame", 2)
        n_target = config.n_averages

        log.info(f"chopper_2x2 cycle start: n_target={n_target} pairs, spf={spf}")

        # Drain stale frames AND their matching tags together to stay in sync.
        # Reading frames first, then exactly n_frames * spf tags ensures the
        # drain doesn't desync the tag-to-frame alignment.
        drain_frames, n_drain = camera.get_buffered_frames()
        if n_drain > 0:
            try:
                self._phase_reader.read_tags(n_drain * spf)
            except Exception:
                self._phase_reader.drain()  # fallback if exact read fails

        # Running accumulators
        sum_on: Optional[np.ndarray] = None
        sum_off: Optional[np.ndarray] = None
        n_on = 0
        n_off = 0
        n_discarded_total = 0
        n_frames_total = 0
        best_offset = 0
        offset_detected = False

        t0 = time.perf_counter()
        empty_reads = 0
        max_empty_reads = 10  # give up after 10 consecutive empty reads (~10s)
        hb_interval = float(getattr(self, "_heartbeat_interval", 1.0))
        last_hb = t0

        while min(n_on, n_off) < n_target:
            now = time.perf_counter()
            if now - last_hb >= hb_interval:
                log.info(
                    f"chopper_2x2 heartbeat: frames={n_frames_total} "
                    f"ON={n_on} OFF={n_off} empty={empty_reads} "
                    f"elapsed={now - t0:.1f}s"
                )
                last_hb = now
            if abort_check is not None and abort_check():
                raise RuntimeError("chopper_2x2: acquisition aborted")

            # Try reading available frames first; sleep only if buffer is empty
            frames, n_chunk = camera.get_buffered_frames()
            if n_chunk == 0:
                empty_reads += 1
                if empty_reads >= max_empty_reads:
                    raise RuntimeError(
                        f"chopper_2x2: no frames after {max_empty_reads} reads "
                        f"({n_frames_total} total, {n_on} ON, {n_off} OFF) — "
                        f"check trigger"
                    )
                time.sleep(self._read_interval)
                continue
            empty_reads = 0

            tags = self._phase_reader.read_tags(n_chunk * spf)
            n_frames_total += n_chunk

            # Detect tag-to-frame alignment offset on first chunk
            if not offset_detected and len(tags) >= spf:
                best_matched = -1
                for offset in range(spf):
                    usable = (len(tags) - offset) // spf
                    if usable < 1:
                        continue
                    grp = tags[offset:offset + usable * spf].reshape(usable, spf)
                    matched = int((grp == grp[:, :1]).all(axis=1).sum())
                    if matched > best_matched:
                        best_matched = matched
                        best_offset = offset
                offset_detected = True

            # Group tags and classify frames
            usable = min(n_chunk, (len(tags) - best_offset) // spf)
            if usable < 1:
                continue

            tag_groups = tags[best_offset:best_offset + usable * spf].reshape(
                usable, spf
            )
            chunk_frames = frames[:usable]

            matched_mask = (tag_groups == tag_groups[:, :1]).all(axis=1)
            n_discarded_total += int(usable - matched_mask.sum())
            matched_frames = chunk_frames[matched_mask]
            matched_tags = tag_groups[matched_mask, 0]

            on_frames = matched_frames[matched_tags == 1]
            off_frames = matched_frames[matched_tags == 0]

            # Accumulate running sums
            if len(on_frames) > 0:
                if sum_on is None:
                    sum_on = on_frames.sum(axis=0).astype(np.float64)
                else:
                    sum_on += on_frames.sum(axis=0)
                n_on += len(on_frames)

            if len(off_frames) > 0:
                if sum_off is None:
                    sum_off = off_frames.sum(axis=0).astype(np.float64)
                else:
                    sum_off += off_frames.sum(axis=0)
                n_off += len(off_frames)

            # Safety: bail if we've read enough frames but have no valid pairs.
            # In ideal conditions we need ~2*n_target frames, so if we've read
            # 3x that with 0 pairs, the tags are clearly broken.
            min_check = max(n_target * 3, 100)
            if n_frames_total >= min_check and min(n_on, n_off) == 0:
                raise RuntimeError(
                    f"chopper_2x2: {n_frames_total} frames read, "
                    f"{n_discarded_total} discarded, {n_on} ON, {n_off} OFF — "
                    f"no valid pairs (check chopper phase sync)"
                )

            if progress_callback is not None:
                elapsed = time.perf_counter() - t0
                progress_callback(min(n_on, n_off), n_target, elapsed)

            # Brief pause before next read to avoid tight-looping on real hardware.
            # Real cameras need time to accumulate frames; mocks return instantly
            # and exit the loop quickly via the safety check above.
            if min(n_on, n_off) < n_target:
                time.sleep(self._read_interval)

        # Compute final result from running sums
        if n_on == 0 or n_off == 0 or sum_on is None or sum_off is None:
            raise RuntimeError(
                f"chopper_2x2: {n_frames_total} frames, "
                f"{n_on} ON, {n_off} OFF — no valid pairs"
            )

        mean_on = sum_on / n_on
        mean_off = sum_off / n_off

        if dark is not None:
            mean_on = mean_on - dark
            mean_off = mean_off - dark

        ref_safe = np.where(mean_off == 0, 1.0, mean_off)
        delta = (mean_on - mean_off) / ref_safe

        n_pairs = min(n_on, n_off)
        log.info(
            f"chopper_2x2 incremental: {n_frames_total} frames total, "
            f"{n_discarded_total} discarded, {n_on} ON, {n_off} OFF, "
            f"{n_pairs} pairs  |  "
            f"ON mean={mean_on.mean():.1f}, OFF mean={mean_off.mean():.1f}"
        )

        last_acquisition_stats.update({
            "pump_mean": mean_on,
            "pump_std": np.zeros_like(mean_on),  # not tracked incrementally
            "ref_mean": mean_off,
            "ref_std": np.zeros_like(mean_off),
            "delta_std": np.zeros_like(delta),
            "n_on": n_on,
            "n_off": n_off,
        })

        if raw_callback is not None:
            raw_callback(mean_on, mean_off, n_pairs, n_discarded_total, n_frames_total)

        return delta
