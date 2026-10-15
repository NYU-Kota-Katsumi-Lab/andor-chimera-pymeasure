"""Camera settings must reach the camera before non-TA acquisitions.

Regression: the main-window acquire button and the experiment queue called
acquire_fvb() without apply_camera_settings(), so the camera stayed on the
SDK default (EM) amplifier while metadata recorded the UI value. EM readout
is mirrored on the DU970P, so spectra were saved reversed.
"""

from __future__ import annotations

import os

os.environ["ANDOR_MOCK"] = "1"

from unittest.mock import patch

import pytest


@pytest.fixture
def hw_manager(qt_app, hardware_manager, wait_for):
    completed = []
    hardware_manager.initialize(on_complete=lambda: completed.append(True))
    wait_for(lambda: len(completed) > 0)
    return hardware_manager


def _record_calls(camera, calls):
    """Wrap camera methods so their call order is recorded."""
    orig_apply = camera.apply_camera_settings
    orig_fvb = camera.acquire_fvb

    def apply(settings):
        calls.append(("apply", dict(settings)))
        return orig_apply(settings)

    def fvb(*args, **kwargs):
        calls.append(("acquire_fvb", None))
        return orig_fvb(*args, **kwargs)

    return (
        patch.object(camera, "apply_camera_settings", side_effect=apply),
        patch.object(camera, "acquire_fvb", side_effect=fvb),
    )


class TestExperimentQueueAppliesSettings:
    def test_settings_applied_before_spectrum(self, hw_manager):
        from andor_qt.core.experiment_queue import ExperimentQueueRunner
        from andor_qt.procedures import SpectrumProcedure

        proc = SpectrumProcedure()
        proc.exposure_time = 0.01
        proc.center_wavelength = 500.0
        proc.grating = 1
        proc.hbin = 1
        proc.num_accumulations = 1
        proc.amplifier_type = 1
        proc.vs_speed_index = 1
        proc.hs_speed_index = 0
        proc.em_gain = 1
        proc.preamp_gain_index = 0

        runner = ExperimentQueueRunner(hw_manager)
        calls = []
        p_apply, p_fvb = _record_calls(hw_manager.camera, calls)
        with p_apply, p_fvb:
            runner._execute_procedure(proc)

        names = [c[0] for c in calls]
        assert "apply" in names, "camera settings were never applied"
        assert names.index("apply") < names.index("acquire_fvb")
        applied = next(c[1] for c in calls if c[0] == "apply")
        assert applied["amplifier_type"] == 1
        assert applied["vs_speed_index"] == 1


class TestMainWindowAppliesSettings:
    def test_settings_applied_before_button_acquisition(self, hw_manager, wait_for):
        from andor_qt.windows.main_window import AndorSpectrometerWindow

        window = AndorSpectrometerWindow()
        cam_settings = {
            "trigger_mode": "internal",
            "exposure_time": 0.01,
            "vs_speed_index": 1,
            "hs_speed_index": 0,
            "amplifier_type": 1,
            "em_gain": 1,
            "preamp_gain_index": 0,
            "read_area_mode": "full",
        }
        params = {
            "exposure_time": 0.01,
            "hbin": 1,
            "read_mode": "fvb",
            "num_accumulations": 1,
            "camera_settings": cam_settings,
        }

        calls = []
        p_apply, p_fvb = _record_calls(hw_manager.camera, calls)
        with p_apply, p_fvb, patch.object(
            window._inputs_widget, "get_parameters", return_value=params
        ), patch.object(window, "_save_data"):
            window._on_queue_clicked()
            assert wait_for(lambda: any(c[0] == "acquire_fvb" for c in calls))

        names = [c[0] for c in calls]
        assert "apply" in names, "camera settings were never applied"
        assert names.index("apply") < names.index("acquire_fvb")
        applied = next(c[1] for c in calls if c[0] == "apply")
        assert applied["amplifier_type"] == 1
