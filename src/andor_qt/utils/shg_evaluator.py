"""Recommend EM vs conventional amplifier for a given signal level.

Given an observed peak signal (electrons per pixel per frame) and the camera
HS-speed setting, compare per-frame SNR under both amplifier modes and return
the recommended mode plus diagnostic numbers.

Per-frame SNR model (shot-noise + read noise, no dark):

- Conventional: ``SNR_conv = S / sqrt(S + RN^2)``
- EM amplifier with excess-noise factor F (typically sqrt(2) at high gain):
  ``SNR_em ≈ S / sqrt(F^2 * S + (RN/G)^2)`` where G is the EM gain. For
  G >> RN the (RN/G)^2 term vanishes, giving ``SNR_em ≈ sqrt(S) / F``.

EM wins when ``SNR_em > SNR_conv``. With F = sqrt(2), the crossover signal
is ``S_x = RN^2``: below it, the read-noise penalty of conventional readout
exceeds the excess-noise penalty of EM amplification.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class AmpRecommendation:
    """Result of an amplifier recommendation comparison.

    Attributes:
        recommended_amp: 0 = EM, 1 = conventional (matches SDK convention).
        snr_conv: SNR under conventional amplifier at this signal level.
        snr_em: SNR under EM amplifier at this signal level (in the high-gain
            limit, so RN/G is neglected).
        crossover_signal_e: Signal in electrons at which the two modes tie.
        signal_e: Echo of the input signal.
        read_noise_e: Echo of the input read noise.
        excess_noise_factor_em: Echo of the EM excess-noise factor used.
        verdict: Human-readable one-line summary.
    """

    recommended_amp: int
    snr_conv: float
    snr_em: float
    crossover_signal_e: float
    signal_e: float
    read_noise_e: float
    excess_noise_factor_em: float
    verdict: str


# ---------------------------------------------------------------------------
# DU970P-typical defaults (approximate; for accuracy substitute measured PTC).
# Keys: (amplifier_type, hs_speed_index, preamp_gain_index)
# Values: electrons per ADU at preamp=0 (lowest gain) reference; higher preamp
# divides this value. These come from Andor's published spec range for the
# camera family and are accurate to ~30% — measure your own PTC for better.
# ---------------------------------------------------------------------------
_DU970P_ADU_PER_ELECTRON = {
    # Conventional amplifier
    (1, 0, 0): 3.5,    # 3 MHz, preamp x1
    (1, 0, 1): 1.7,    # 3 MHz, preamp x2
    (1, 0, 2): 0.85,   # 3 MHz, preamp x4
    (1, 1, 0): 2.5,    # 1 MHz, preamp x1
    (1, 1, 1): 1.25,   # 1 MHz, preamp x2
    (1, 1, 2): 0.6,    # 1 MHz, preamp x4
    (1, 2, 0): 1.5,    # 50 kHz, preamp x1
    (1, 2, 1): 0.75,   # 50 kHz, preamp x2
    (1, 2, 2): 0.4,    # 50 kHz, preamp x4
    # EM amplifier (typical values; actual depends on EM gain setting)
    (0, 0, 0): 25.0,   # 3 MHz, preamp x1 (EM amp has different ADC scaling)
    (0, 1, 0): 17.0,   # 1 MHz, preamp x1
}

_DU970P_READ_NOISE_E = {
    # Conventional amplifier
    (1, 0): 12.0,   # 3 MHz
    (1, 1): 6.0,    # 1 MHz
    (1, 2): 4.0,    # 50 kHz
    # EM amplifier (raw before EM gain divides it; effective noise ≈ this / EM_gain)
    (0, 0): 15.0,
    (0, 1): 8.0,
}

# Fallback values for unknown (amp, hs_speed, preamp_gain) combinations.
_FALLBACK_ADU_PER_ELECTRON = 3.0
_FALLBACK_READ_NOISE_E = 10.0


def default_adu_per_electron(
    amplifier_type: int,
    hs_speed_index: int,
    preamp_gain_index: int,
) -> float:
    """Return a typical e-/ADU conversion factor for the DU970P.

    These are approximate per-camera-family values; for accuracy substitute
    a measured photon-transfer-curve K factor.
    """
    key = (amplifier_type, hs_speed_index, preamp_gain_index)
    return _DU970P_ADU_PER_ELECTRON.get(key, _FALLBACK_ADU_PER_ELECTRON)


def default_read_noise_e(
    amplifier_type: int,
    hs_speed_index: int,
) -> float:
    """Return the typical read noise in electrons rms for the DU970P at the
    given amplifier / HS-speed combination."""
    return _DU970P_READ_NOISE_E.get(
        (amplifier_type, hs_speed_index), _FALLBACK_READ_NOISE_E
    )


def recommend_amplifier_for_shg(
    signal_e: float,
    read_noise_e: float,
    excess_noise_factor_em: float = math.sqrt(2.0),
) -> AmpRecommendation:
    """Compare per-frame SNR for EM vs conventional and recommend a mode.

    Args:
        signal_e: Observed peak signal in electrons per pixel per frame.
        read_noise_e: Read noise of the relevant readout path (electrons rms).
        excess_noise_factor_em: EM amplifier excess-noise factor (typically
            sqrt(2) at high gain; can be larger for very high gain).

    Returns:
        ``AmpRecommendation`` with both SNRs, the crossover signal, and a
        one-line verdict.

    Raises:
        ValueError: If signal is negative or read noise is non-positive.
    """
    if signal_e < 0:
        raise ValueError(f"signal_e must be non-negative, got {signal_e}")
    if read_noise_e <= 0:
        raise ValueError(f"read_noise_e must be positive, got {read_noise_e}")

    F = excess_noise_factor_em
    rn = read_noise_e

    # Crossover signal where SNR_em == SNR_conv (high-gain limit):
    #   S / sqrt(S + RN^2) == sqrt(S) / F
    # =>  F^2 * S == S + RN^2
    # =>  S * (F^2 - 1) == RN^2
    # =>  S_x == RN^2 / (F^2 - 1)
    fsq_m1 = F * F - 1.0
    if fsq_m1 <= 0:
        # Pathological: F<=1 means EM has no excess noise penalty — always wins.
        crossover = float("inf")
    else:
        crossover = (rn * rn) / fsq_m1

    if signal_e == 0:
        snr_conv = 0.0
        snr_em = 0.0
    else:
        snr_conv = signal_e / math.sqrt(signal_e + rn * rn)
        snr_em = math.sqrt(signal_e) / F

    # Below crossover: EM. Above: conventional. At zero signal, prefer EM
    # (it's the read-noise-limited regime).
    recommended = 0 if snr_em >= snr_conv else 1

    if signal_e == 0:
        verdict = (
            f"signal = 0 e-/pixel: no signal to discriminate; EM is the "
            f"right starting point for low-light work (crossover at "
            f"S = {crossover:.0f} e-)."
        )
    elif recommended == 0:
        verdict = (
            f"signal = {signal_e:.1f} e-/pixel is BELOW crossover "
            f"S_x = {crossover:.0f} e- (RN={rn:.1f} e-, F={F:.2f}). "
            f"EM amplifier recommended — SNR_em = {snr_em:.2f} vs "
            f"SNR_conv = {snr_conv:.2f}."
        )
    else:
        verdict = (
            f"signal = {signal_e:.1f} e-/pixel is ABOVE crossover "
            f"S_x = {crossover:.0f} e- (RN={rn:.1f} e-, F={F:.2f}). "
            f"Conventional amplifier recommended — SNR_conv = {snr_conv:.2f} "
            f"vs SNR_em = {snr_em:.2f}."
        )

    return AmpRecommendation(
        recommended_amp=recommended,
        snr_conv=snr_conv,
        snr_em=snr_em,
        crossover_signal_e=crossover,
        signal_e=signal_e,
        read_noise_e=rn,
        excess_noise_factor_em=F,
        verdict=verdict,
    )
