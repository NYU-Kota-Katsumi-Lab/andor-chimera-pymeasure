"""EMCCD health check: verify EM gain mode 3 and gain-scaling on dark.

Block all light to the camera (close spectrometer slit / lens cap) BEFORE
running this. The script:

1. Initializes the camera and reports ``em_gain_mode`` and ``em_gain_range``
   — this confirms ``SetEMGainMode(3)`` took during ``initialize()``.
2. Selects the conventional amplifier and takes a dark frame at baseline
   (the "amp working" reference).
3. Switches to the EM amplifier and takes dark frames at a sweep of EM gain
   values, reporting mean and std per frame.
4. The mean dark counts should grow ROUGHLY linearly with EM gain because
   dark current and clock-induced charge both get multiplied. A flat curve
   (no growth with gain) is the smoking gun for a dead gain register.

Uses internal trigger so no SDG / chopper is required. The cooler state is
not touched — verify the gain register works at whatever temperature the
camera currently holds.

Usage::

    uv run python scripts/emccd_health_check.py
    uv run python scripts/emccd_health_check.py --gains 1,10,50,100,200
    uv run python scripts/emccd_health_check.py --frames 50 --exposure 0.1
"""

from __future__ import annotations

import argparse
import atexit
import logging
import os
import signal
import sys
import time
from pathlib import Path

import numpy as np

# Force UTF-8 stdout so degree signs etc. don't crash on cp1252.
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

os.environ["ANDOR_MOCK"] = ""

# --- Bulletproof camera shutdown (per scripts/chopper_phase_sweep.py) ---
_camera_for_cleanup = None


def _emergency_camera_shutdown():
    global _camera_for_cleanup
    cam = _camera_for_cleanup
    if cam is None:
        return
    _camera_for_cleanup = None
    try:
        cam.abort_acquisition()
    except Exception:
        pass
    try:
        cam.shutdown()
        print("[atexit] camera shutdown OK", flush=True)
    except Exception as e:
        print(f"[atexit] camera shutdown error: {e}", flush=True)


atexit.register(_emergency_camera_shutdown)


def _signal_handler(signum, _frame):
    print(f"\n[!] Received signal {signum}, shutting down...", flush=True)
    raise KeyboardInterrupt()


for _sig_name in ("SIGINT", "SIGBREAK", "SIGTERM"):
    _s = getattr(signal, _sig_name, None)
    if _s is not None:
        try:
            signal.signal(_s, _signal_handler)
        except (ValueError, OSError):
            pass

SDK_PATH = r"C:\Program Files\Andor SDK"


def _take_dark_internal_trigger(cam, n_frames: int, exposure_s: float) -> tuple[float, float, np.ndarray]:
    """Acquire ``n_frames`` dark spectra in internal trigger and return mean, std, frames.

    Uses the camera's existing FVB / hbin setting from apply_camera_settings.
    """
    cam.set_exposure(exposure_s)
    spectra = []
    for _ in range(n_frames):
        spec = cam.get_spectrum()
        spectra.append(np.asarray(spec, dtype=np.float64))
    arr = np.array(spectra)
    return float(arr.mean()), float(arr.std()), arr


def _cool_camera(cam, target_c: int, max_wait_s: float, tol_c: float = 2.0) -> bool:
    """Turn cooler on and wait until ``temperature <= target_c + tol_c``.

    Returns True if target is reached within ``max_wait_s``, False on timeout.
    Prints progress every ~10 seconds so the user can see something is happening.
    """
    current = cam.temperature
    print(f"  current temperature = {current:+.1f} C, target = {target_c:+d} C")
    if current <= target_c + tol_c:
        print("  already at target — no cooling needed")
        return True
    print(f"  enabling cooler (max wait {max_wait_s:.0f} s)...")
    cam.cooler_on(target=target_c)
    t0 = time.time()
    last_print = t0
    while time.time() - t0 < max_wait_s:
        now = time.time()
        temp = cam.temperature
        if now - last_print >= 10.0:
            elapsed = now - t0
            print(f"    +{elapsed:6.0f} s : T = {temp:+6.1f} C")
            last_print = now
        if temp <= target_c + tol_c:
            elapsed = time.time() - t0
            print(f"  reached target after {elapsed:.0f} s : T = {temp:+.1f} C")
            return True
        time.sleep(2.0)
    print(f"  TIMEOUT: cooler did not reach {target_c:+d} C within {max_wait_s:.0f} s")
    print(f"  current = {cam.temperature:+.1f} C — proceeding anyway")
    return False


def _interpret_gain_sweep(
    em_mode: int,
    temperature_c: float,
    gain_pairs: list[tuple[int, float]],
    std_pairs: list[tuple[int, float]] | None = None,
    conv_mean: float | None = None,
) -> str:
    """Return a verdict string for the gain sweep.

    DAC mode 0 (``em_mode=0``) is highly non-linear: low gain codes barely
    amplify; the curve climbs steeply above ~DAC 100. Real EM Gain mode 3
    is roughly linear in physical multiplication factor.

    Temperature matters too — EM gain rises sharply as the sensor cools.
    But at very cold temperatures with no illumination, dark current drops
    below the level where it can be visibly amplified. In that regime, the
    test is inconclusive: the registers may be working perfectly but there's
    nothing to amplify. Look at the std growth instead, which traces shot
    noise on whatever residual charge is being clocked through.
    """
    if len(gain_pairs) < 2:
        return "(too few gain points to interpret)"
    g_lo, m_lo = gain_pairs[0]
    g_hi, m_hi = gain_pairs[-1]
    ratio_obs = m_hi / max(m_lo, 1e-9)
    is_cold = temperature_c < -40.0
    label = f"Mean(gain={g_hi}) / Mean(gain={g_lo}) = {ratio_obs:.2f}"

    # Std-based amplification signature: even when mean is dominated by bias,
    # variance traces the amplified shot noise.
    std_signal = ""
    if std_pairs is not None and len(std_pairs) >= 2:
        _, s_lo = std_pairs[0]
        _, s_hi = std_pairs[-1]
        std_ratio = s_hi / max(s_lo, 1e-9)
        std_signal = f"  std ratio = {std_ratio:.2f}"

    if em_mode == 3:
        # Linear scaling expected (when there's signal)
        ratio_expected = g_hi / max(g_lo, 1e-9)
        if ratio_obs > 0.5 * ratio_expected:
            return f"{label}; expected ~{ratio_expected:.0f}× (linear). PASS.{std_signal}"
        if ratio_obs > 1.5:
            return (f"{label}; expected ~{ratio_expected:.0f}× (linear). "
                    f"PARTIAL — register may be aged or saturated.{std_signal}")
        return f"{label}; expected ~{ratio_expected:.0f}×. FAIL — gain register dead?{std_signal}"

    # DAC mode 0: non-linear physical response, T-dependent
    if is_cold:
        # Check whether there is any signal to amplify in the first place.
        # If the conv-amp baseline is dominated by the bias offset (low ADU)
        # and our mean barely differs from it, dark current is too low for
        # this exposure to show amplification.
        signal_too_low = (
            conv_mean is not None
            and conv_mean < 500
            and ratio_obs < 1.1
        )
        if signal_too_low:
            return (
                f"{label} at {temperature_c:+.0f} C. INCONCLUSIVE — at cold "
                f"temperature with no light the dark current is too low to "
                f"reveal amplification (conv baseline {conv_mean:.0f} ADU). "
                f"{std_signal.strip()}. Increase --exposure to 1-5 s or "
                f"admit a controlled light source, then re-test."
            )
        if ratio_obs >= 15:
            return f"{label} at {temperature_c:+.0f} C. PASS — strong EM amplification.{std_signal}"
        if ratio_obs >= 3:
            return (f"{label} at {temperature_c:+.0f} C. PARTIAL — growth present "
                    f"but lower than expected at cold temperature.{std_signal}")
        return (f"{label} at {temperature_c:+.0f} C. FAIL — barely any EM "
                f"amplification despite cold operation.{std_signal}")
    # Warm: DAC 200 ≈ x3-6 physical
    if ratio_obs >= 2.5:
        return (f"{label} at {temperature_c:+.0f} C (warm). PASS — gain register "
                f"is alive but operating at low efficiency due to warm sensor. "
                f"Re-test cooled for the real curve.{std_signal}")
    if ratio_obs > 1.3:
        return (f"{label} at {temperature_c:+.0f} C (warm). PARTIAL — small "
                f"growth visible; cool the sensor and re-test.{std_signal}")
    return (f"{label} at {temperature_c:+.0f} C (warm). FAIL — counts barely "
            f"change with gain; either amp didn't switch or register is dead.{std_signal}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--gains", default="1,5,10,25,50,100,200",
                        help="Comma-separated EM gain values to sweep (default 1,5,10,25,50,100,200)")
    parser.add_argument("--frames", type=int, default=20,
                        help="Frames to average per gain step (default 20)")
    parser.add_argument("--exposure", type=float, default=0.1,
                        help="Exposure per frame in seconds (default 0.1 s)")
    parser.add_argument("--vs-speed", type=int, default=1,
                        help="VS speed index (default 1 = 9.8 us, recommended for EM)")
    parser.add_argument("--hs-speed", type=int, default=1,
                        help="HS speed index for EM amp (default 1 = 1 MHz)")
    parser.add_argument("--preamp-gain", type=int, default=0,
                        help="Pre-amp gain index (default 0 = lowest)")
    parser.add_argument("--hbin", type=int, default=1,
                        help="Horizontal binning (default 1 = full resolution)")
    parser.add_argument("--cooler-target", type=int, default=-65,
                        help="Cooler target temperature in C (default -65). "
                             "Set to 0 or higher to skip cooling.")
    parser.add_argument("--cooler-wait", type=float, default=900.0,
                        help="Max seconds to wait for target temperature (default 900)")
    args = parser.parse_args()

    gain_list = [int(g.strip()) for g in args.gains.split(",") if g.strip()]

    # Quiet noisy debug logs from the camera driver
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    from andor_pymeasure.instruments.andor_camera import AndorCamera

    global _camera_for_cleanup
    print("=" * 70)
    print("EMCCD health check")
    print("=" * 70)
    print()
    print("** Ensure all light is blocked from the camera before continuing **")
    print("  (close spectrometer slit / put lens cap on / dark room)")
    print()
    print("Initializing camera...")
    cam = AndorCamera(sdk_path=SDK_PATH)
    cam.initialize()
    _camera_for_cleanup = cam
    print(f"  serial = {cam._info.serial_number}")
    print(f"  detector = {cam._info.xpixels} x {cam._info.ypixels}")
    print(f"  temperature = {cam.temperature:.1f} C")
    print()

    # --- Step 1: report EM gain mode and range ---
    print("-- EM gain mode --")
    print(f"  em_gain_mode = {cam.em_gain_mode}")
    if cam.em_gain_mode == 3:
        print("  (Real EM Gain — integer = physical multiplication factor)")
    elif cam.em_gain_mode == 0:
        print("  (8-bit DAC fallback — integer is opaque code 0-255)")
    em_low, em_high = cam.info.em_gain_range
    print(f"  em_gain_range = ({em_low}, {em_high})")
    if cam.em_gain_mode == 3 and em_high > 255:
        print("  WARNING: range high is >255 in mode 3 — unexpected")
    elif cam.em_gain_mode == 3 and (em_low, em_high) == (0, 255):
        print("  WARNING: range looks like DAC codes — mode 3 may not have taken")
    print()

    # --- Probe which EM gain modes the SDK accepts ---
    print("-- EM gain mode capability probe --")
    DRV_SUCCESS = cam._errors.Error_Codes.DRV_SUCCESS
    mode_names = {
        0: "8-bit DAC (0-255)",
        1: "12-bit DAC (0-4095)",
        2: "Linear (12-bit DAC)",
        3: "Real EM Gain",
    }
    supported = []
    for m in (0, 1, 2, 3):
        r = cam._sdk.SetEMGainMode(m)
        mark = "yes" if r == DRV_SUCCESS else f"no (code {r})"
        print(f"  mode {m} ({mode_names[m]:<22}) : {mark}")
        if r == DRV_SUCCESS:
            supported.append(m)
    # Restore to highest available preferred order: 3 > 2 > 0
    for preferred in (3, 2, 0):
        if preferred in supported:
            cam._sdk.SetEMGainMode(preferred)
            cam._em_gain_mode = preferred
            print(f"  -> using mode {preferred} ({mode_names[preferred]})")
            break
    print()

    # --- Firmware / hardware version (best-effort) ---
    print("-- Hardware version --")
    try:
        result = cam._sdk.GetHardwareVersion()
        # Andor SDK Python wrapper returns (ret, pcb, decode, d1, d2, fw_ver, fw_build)
        # We don't know which order — print whatever we get.
        print(f"  GetHardwareVersion -> {result}")
    except Exception as e:
        print(f"  GetHardwareVersion failed: {e}")
    try:
        result = cam._sdk.GetSoftwareVersion()
        print(f"  GetSoftwareVersion -> {result}")
    except Exception as e:
        print(f"  GetSoftwareVersion failed: {e}")
    print()

    # --- Step 1b: cooling ---
    if args.cooler_target < 0:
        print("-- Cooling --")
        _cool_camera(cam, args.cooler_target, args.cooler_wait)
        print()
    else:
        print(f"-- Cooling skipped (target {args.cooler_target}) --")
        print(f"  current temperature = {cam.temperature:+.1f} C")
        print()

    try:
        # --- Step 2: conventional amplifier baseline ---
        print("-- Conventional amplifier baseline --")
        cam.apply_camera_settings({
            "trigger_mode": "internal",
            "exposure_time": args.exposure,
            "vs_speed_index": args.vs_speed,
            "hs_speed_index": args.hs_speed,
            "amplifier_type": 1,  # conventional
            "preamp_gain_index": args.preamp_gain,
            "hbin": args.hbin,
            "vbin": 1,
            "read_area_mode": "full",
        })
        time.sleep(0.2)
        conv_mean, conv_std, _ = _take_dark_internal_trigger(cam, args.frames, args.exposure)
        print(f"  {args.frames} frames @ {args.exposure*1000:.0f} ms : "
              f"mean = {conv_mean:.1f} ADU,  std = {conv_std:.1f} ADU")
        print()

        # --- Step 3: EM amplifier — sweep gain ---
        print("-- EM amplifier — dark mean vs EM gain --")
        print(f"  {'EM gain':>10}  {'mean ADU':>12}  {'std ADU':>12}  {'mean/conv':>12}")
        print(f"  {'-'*10}  {'-'*12}  {'-'*12}  {'-'*12}")

        results = []
        for gain in gain_list:
            # Clamp to valid range
            gain_clamped = max(em_low, min(gain, em_high))
            cam.apply_camera_settings({
                "amplifier_type": 0,    # EM
                "hs_speed_index": args.hs_speed,
                "em_gain": gain_clamped,
                "preamp_gain_index": args.preamp_gain,
                "hbin": args.hbin,
                "vbin": 1,
                "read_area_mode": "full",
            })
            # Verify readback
            actual = cam.get_em_gain()
            time.sleep(0.1)
            mean_v, std_v, _ = _take_dark_internal_trigger(cam, args.frames, args.exposure)
            ratio = mean_v / conv_mean if conv_mean > 0 else float("nan")
            tag = "" if gain_clamped == gain else f" (clamped from {gain})"
            print(f"  {gain_clamped:>10}{tag}  {mean_v:>12.1f}  {std_v:>12.1f}  {ratio:>12.2f}")
            if actual != gain_clamped:
                print(f"    !! readback mismatch: requested {gain_clamped}, got {actual}")
            results.append((gain_clamped, mean_v, std_v))

        print()
        print("-- Interpretation --")
        verdict = _interpret_gain_sweep(
            cam.em_gain_mode,
            cam.temperature,
            [(g, m) for g, m, _ in results],
            std_pairs=[(g, s) for g, _, s in results],
            conv_mean=conv_mean,
        )
        print(f"  {verdict}")

    finally:
        try:
            cam.abort_acquisition()
        except Exception:
            pass
        print()
        print("Shutting down camera...")
        try:
            cam.shutdown()
            print("  camera shutdown OK")
        except Exception as e:
            print(f"  camera shutdown error: {e}")
        _camera_for_cleanup = None

    return 0


if __name__ == "__main__":
    sys.exit(main())
