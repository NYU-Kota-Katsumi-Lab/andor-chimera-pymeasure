"""Tests for the SHG amplifier-recommendation utility."""

from __future__ import annotations

import math

import pytest


class TestRecommendAmplifierForShg:
    """The recommendation compares per-frame SNR under conventional vs EM
    amplifier modes:

    - Conventional: SNR = S / sqrt(S + RN^2)
    - EM (high gain, excess-noise factor F = sqrt(2)): SNR ≈ sqrt(S) / F

    With F = sqrt(2), EM wins when S < RN^2 (the crossover signal).
    """

    def test_low_signal_em_recommended(self):
        from andor_qt.utils.shg_evaluator import recommend_amplifier_for_shg

        # Fast HS read noise ~10 e-. Signal of 5 e- is well below crossover (100 e-).
        result = recommend_amplifier_for_shg(signal_e=5.0, read_noise_e=10.0)
        assert result.recommended_amp == 0  # 0 = EM
        assert result.snr_em > result.snr_conv

    def test_high_signal_conventional_recommended(self):
        from andor_qt.utils.shg_evaluator import recommend_amplifier_for_shg

        # Far above crossover: conventional wins (no excess-noise penalty).
        result = recommend_amplifier_for_shg(signal_e=5000.0, read_noise_e=10.0)
        assert result.recommended_amp == 1  # 1 = conventional
        assert result.snr_conv > result.snr_em

    def test_crossover_at_read_noise_squared(self):
        """For F=sqrt(2), the modes are tied when signal = read_noise^2."""
        from andor_qt.utils.shg_evaluator import recommend_amplifier_for_shg

        rn = 10.0
        crossover = rn * rn  # = 100 e-
        result = recommend_amplifier_for_shg(
            signal_e=crossover,
            read_noise_e=rn,
            excess_noise_factor_em=math.sqrt(2.0),
        )
        # SNRs should be approximately equal at the crossover
        assert result.snr_conv == pytest.approx(result.snr_em, rel=0.05)

    def test_zero_signal_returns_zero_snr(self):
        from andor_qt.utils.shg_evaluator import recommend_amplifier_for_shg

        result = recommend_amplifier_for_shg(signal_e=0.0, read_noise_e=10.0)
        assert result.snr_conv == 0.0
        assert result.snr_em == 0.0
        # With zero signal we still want a deterministic answer; EM is the
        # right answer because it gets less penalised by RN at low signal.
        assert result.recommended_amp == 0

    def test_returns_crossover_for_caller_to_print(self):
        from andor_qt.utils.shg_evaluator import recommend_amplifier_for_shg

        result = recommend_amplifier_for_shg(signal_e=10.0, read_noise_e=5.0)
        # Crossover for RN=5 with F=sqrt(2) is 5^2 = 25 e-
        assert result.crossover_signal_e == pytest.approx(25.0, rel=0.01)

    def test_excess_noise_factor_override(self):
        """A larger excess-noise factor moves the crossover up — EM is worse,
        conventional wins at signals where it previously didn't."""
        from andor_qt.utils.shg_evaluator import recommend_amplifier_for_shg

        # Signal slightly above default crossover (RN^2) — EM normally edged out
        # with F=sqrt(2). With a worse F (say 2.0), EM should lose more decisively.
        rn = 10.0
        sig = rn * rn * 1.1  # just above crossover
        worse_F = recommend_amplifier_for_shg(
            signal_e=sig, read_noise_e=rn, excess_noise_factor_em=2.0
        )
        better_F = recommend_amplifier_for_shg(
            signal_e=sig, read_noise_e=rn, excess_noise_factor_em=math.sqrt(2.0)
        )
        # Worse excess noise → lower EM SNR
        assert worse_F.snr_em < better_F.snr_em

    def test_verdict_string_contains_signal_and_crossover(self):
        from andor_qt.utils.shg_evaluator import recommend_amplifier_for_shg

        result = recommend_amplifier_for_shg(signal_e=50.0, read_noise_e=10.0)
        text = result.verdict.lower()
        assert "50" in text  # signal value
        assert "100" in text or "1e+02" in text  # crossover

    def test_negative_signal_raises(self):
        from andor_qt.utils.shg_evaluator import recommend_amplifier_for_shg

        with pytest.raises(ValueError, match="non-negative"):
            recommend_amplifier_for_shg(signal_e=-5.0, read_noise_e=10.0)

    def test_zero_read_noise_raises(self):
        from andor_qt.utils.shg_evaluator import recommend_amplifier_for_shg

        with pytest.raises(ValueError, match="positive"):
            recommend_amplifier_for_shg(signal_e=10.0, read_noise_e=0.0)


class TestAduToElectrons:
    """The script needs to convert observed peak ADU into electrons. The
    conversion factor depends on amplifier, HS speed, and preamp gain — we
    use a documented lookup with fallbacks. The lookup itself is editable
    so a user can substitute measured PTC values."""

    def test_default_lookup_has_du970p_values(self):
        from andor_qt.utils.shg_evaluator import default_adu_per_electron

        # Conv amp, 3 MHz, preamp 0 should be in the ~2-4 range
        val = default_adu_per_electron(amplifier_type=1, hs_speed_index=0, preamp_gain_index=0)
        assert 1.0 < val < 6.0

    def test_unknown_combination_returns_fallback(self):
        from andor_qt.utils.shg_evaluator import default_adu_per_electron

        # Way out of range — should still return a positive fallback, not raise
        val = default_adu_per_electron(amplifier_type=1, hs_speed_index=99, preamp_gain_index=99)
        assert val > 0

    def test_default_read_noise_has_du970p_values(self):
        from andor_qt.utils.shg_evaluator import default_read_noise_e

        # 3 MHz fast readout should have higher noise than 50 kHz slow scan
        fast = default_read_noise_e(amplifier_type=1, hs_speed_index=0)
        slow = default_read_noise_e(amplifier_type=1, hs_speed_index=2)
        assert fast > slow
        # Sanity: noise should be in single-to-double-digits of electrons
        assert 1.0 < slow < 50.0
        assert 1.0 < fast < 50.0
