"""Externally triggered single acquisitions must not wait forever.

Since main-window acquisitions apply the UI camera settings, a "fast_external"
or "external" trigger mode with no laser triggers present would block in
WaitForAcquisition until Abort. AndorCamera now waits a bounded time and
raises a clear TimeoutError instead.
"""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture
def cam(initialized_camera):
    return initialized_camera


class TestExternalTriggerTimeout:
    def test_internal_trigger_acquires(self, cam):
        cam.apply_camera_settings({"trigger_mode": "internal"})
        cam._sdk._state.external_trigger_present = False  # irrelevant for internal
        assert isinstance(cam.acquire_fvb(), np.ndarray)

    @pytest.mark.parametrize("mode", ["external", "fast_external"])
    def test_external_with_triggers_acquires(self, cam, mode):
        cam.apply_camera_settings({"trigger_mode": mode})
        assert len(cam.acquire_fvb()) == cam.xpixels

    @pytest.mark.parametrize("mode", ["external", "fast_external"])
    def test_external_without_trigger_times_out(self, cam, mode):
        cam.apply_camera_settings({"trigger_mode": mode})
        cam._sdk._state.external_trigger_present = False
        with pytest.raises(TimeoutError, match="trigger"):
            cam.acquire_fvb()
        assert cam._sdk._state.acquiring is False  # acquisition was aborted

    def test_image_without_trigger_times_out(self, cam):
        cam.apply_camera_settings({"trigger_mode": "external"})
        cam._sdk._state.external_trigger_present = False
        with pytest.raises(TimeoutError, match="trigger"):
            cam.acquire_image()

    def test_timeout_includes_exposure(self, cam):
        cam.apply_camera_settings({"trigger_mode": "external", "exposure_time": 2.0})
        cam._sdk._state.external_trigger_present = False
        with pytest.raises(TimeoutError) as exc:
            cam.acquire_fvb()
        expected = cam.EXTERNAL_TRIGGER_TIMEOUT_S + 2.0
        assert f"{expected:.1f} s" in str(exc.value)
