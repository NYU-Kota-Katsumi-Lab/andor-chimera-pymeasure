"""Binned wavelength axes must give the wavelength at each bin's centre.

Regression: callers asked GetCalibration for N/b pixels of width b*w. On the
Kymera 193i the SDK then labels bin k with unbinned pixel b*k + b - 1, i.e.
(b-1)/2 px to the red of the bin centre (+0.128 nm for hbin 8 at 400 nm,
measured with argon 2026-10-15).
"""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture
def spec(initialized_spectrograph):
    return initialized_spectrograph


class TestGetBinnedCalibration:
    def test_hbin1_matches_unbinned(self, spec):
        np.testing.assert_allclose(
            spec.get_binned_calibration(1600, 16.0, 1), spec.get_calibration(1600, 16.0)
        )

    @pytest.mark.parametrize("hbin", [2, 4, 8])
    def test_binned_is_mean_of_unbinned_bin(self, spec, hbin):
        unbinned = spec.get_calibration(1600, 16.0)
        binned = spec.get_binned_calibration(1600, 16.0, hbin)
        assert binned.shape == (1600 // hbin,)
        np.testing.assert_allclose(binned, unbinned.reshape(-1, hbin).mean(axis=1))

    def test_invalid_hbin_raises(self, spec):
        with pytest.raises(ValueError, match="hbin"):
            spec.get_binned_calibration(1600, 16.0, 7)
