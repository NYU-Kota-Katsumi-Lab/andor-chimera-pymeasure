"""The acquire button must fail cleanly, not hang, when no external trigger arrives."""

from __future__ import annotations

import os

os.environ["ANDOR_MOCK"] = "1"

from unittest.mock import patch

from PySide6.QtWidgets import QApplication


def test_button_reports_missing_trigger(qt_app, hardware_manager, wait_for):
    completed = []
    hardware_manager.initialize(on_complete=lambda: completed.append(True))
    wait_for(lambda: len(completed) > 0)

    from andor_qt.windows.main_window import AndorSpectrometerWindow

    window = AndorSpectrometerWindow()
    params = {
        "exposure_time": 0.01, "hbin": 1, "read_mode": "fvb", "num_accumulations": 1,
        "camera_settings": {"trigger_mode": "fast_external", "amplifier_type": 1,
                            "exposure_time": 0.01},
    }
    state = hardware_manager.camera._sdk._state
    failures = []
    window._acq_signals.failed.connect(lambda exp_id, msg: failures.append(msg))
    state.external_trigger_present = False
    try:
        with patch.object(window._inputs_widget, "get_parameters", return_value=params), \
                patch.object(window, "_save_data"):
            window._on_queue_clicked()
            # failed is emitted from the worker thread; process events while waiting
            assert wait_for(lambda: QApplication.processEvents() or len(failures) > 0,
                            timeout=15.0)
    finally:
        state.external_trigger_present = True
        hardware_manager.camera.apply_camera_settings({"trigger_mode": "internal"})

    assert "trigger" in failures[0].lower()
