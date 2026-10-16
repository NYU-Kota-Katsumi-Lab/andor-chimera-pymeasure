"""Amplifier-dependent readout direction on the Newton DU970P.

The EM and conventional output nodes sit at opposite ends of the serial
register, so the EM amplifier reads the CCD right-to-left
(IsReadoutFlippedByAmplifier(0) == 1 on hardware, verified 2026-10-15 with an
argon lamp). AndorCamera must return data in left-to-right (blue -> red)
order regardless of amplifier, matching ATSpectrograph GetCalibration.
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

DRV_SUCCESS = 20002


@pytest.fixture
def cam(initialized_camera):
    return initialized_camera


def _ramp_readout(n_values):
    """GetImages16 stand-in returning 0..n-1 in raw readout order."""

    def fake(first, last, size):
        return (DRV_SUCCESS, list(range(size)), first, last)

    return fake


class TestMockReportsDU970POrientation:
    def test_em_amplifier_is_flipped(self, cam):
        assert cam._sdk.IsReadoutFlippedByAmplifier(0) == (DRV_SUCCESS, 1)

    def test_conventional_amplifier_is_not_flipped(self, cam):
        assert cam._sdk.IsReadoutFlippedByAmplifier(1) == (DRV_SUCCESS, 0)


class TestReadoutFlippedTracking:
    def test_default_amplifier_after_initialize_is_flipped(self, cam):
        # SDK default output amplifier after Initialize is EM (0)
        assert cam.readout_flipped is True

    def test_conventional_via_apply_camera_settings(self, cam):
        cam.apply_camera_settings({"amplifier_type": 1})
        assert cam.readout_flipped is False

    def test_em_via_set_amplifier(self, cam):
        cam.apply_camera_settings({"amplifier_type": 1})
        cam.set_amplifier(0)
        assert cam.readout_flipped is True

    def test_query_failure_assumes_not_flipped(self, cam):
        with patch.object(cam._sdk, "IsReadoutFlippedByAmplifier", return_value=(20991, 0)):
            cam.set_amplifier(0)
        assert cam.readout_flipped is False


class TestReadoutMetadata:
    def test_after_initialize_reports_em_corrected(self, cam):
        assert cam.readout_metadata() == {
            "amplifier_type_applied": 0,
            "readout_flip_corrected": 1,
            "readout_orientation": "short_wavelength_first",
        }

    def test_after_conventional_reports_uncorrected(self, cam):
        cam.apply_camera_settings({"amplifier_type": 1})
        meta = cam.readout_metadata()
        assert meta["amplifier_type_applied"] == 1
        assert meta["readout_flip_corrected"] == 0

    def test_settings_without_amplifier_keep_previous(self, cam):
        cam.set_amplifier(1)
        cam.apply_camera_settings({"hbin": 2})
        assert cam.readout_metadata()["amplifier_type_applied"] == 1


class TestAcquireFvbOrientation:
    def test_flipped_amplifier_reverses_spectrum(self, cam):
        cam.set_amplifier(0)
        with patch.object(cam._sdk, "GetImages16", side_effect=_ramp_readout(cam.xpixels)):
            data = cam.acquire_fvb()
        np.testing.assert_array_equal(data, np.arange(cam.xpixels)[::-1])

    def test_unflipped_amplifier_keeps_order(self, cam):
        cam.set_amplifier(1)
        with patch.object(cam._sdk, "GetImages16", side_effect=_ramp_readout(cam.xpixels)):
            data = cam.acquire_fvb()
        np.testing.assert_array_equal(data, np.arange(cam.xpixels))

    def test_flipped_with_hbin(self, cam):
        cam.set_amplifier(0)
        n = cam.xpixels // 8
        with patch.object(cam._sdk, "GetImages16", side_effect=_ramp_readout(n)):
            data = cam.acquire_fvb(hbin=8)
        np.testing.assert_array_equal(data, np.arange(n)[::-1])


class TestBufferedFramesOrientation:
    def _read_two_frames(self, cam):
        cam.start_run_till_abort()
        with patch.object(cam._sdk, "GetNumberNewImages", return_value=(DRV_SUCCESS, 1, 2)), \
                patch.object(cam._sdk, "GetImages16", side_effect=_ramp_readout(0)):
            frames, n = cam.get_buffered_frames()
        cam.abort_acquisition()
        return frames, n

    def test_flipped_reverses_each_frame_not_frame_order(self, cam):
        cam.set_amplifier(0)
        frames, n = self._read_two_frames(cam)
        raw = np.arange(2 * cam.xpixels).reshape(2, cam.xpixels)
        assert n == 2
        np.testing.assert_array_equal(frames, raw[:, ::-1])

    def test_unflipped_keeps_frames(self, cam):
        cam.set_amplifier(1)
        frames, _ = self._read_two_frames(cam)
        raw = np.arange(2 * cam.xpixels).reshape(2, cam.xpixels)
        np.testing.assert_array_equal(frames, raw)


class TestAcquireImageOrientation:
    def test_flipped_reverses_columns(self, cam):
        cam.set_amplifier(0)
        size = cam.xpixels * cam.ypixels
        with patch.object(cam._sdk, "GetImages16", side_effect=_ramp_readout(size)):
            img = cam.acquire_image()
        raw = np.arange(size).reshape(cam.ypixels, cam.xpixels)
        np.testing.assert_array_equal(img, raw[:, ::-1])
