"""Saved data must record the readout state the camera actually used.

Before 2026-10-15 the .meta.json recorded the UI amplifier while the camera
could still be on a different one. Files now carry amplifier_type_applied,
readout_flip_corrected and readout_orientation taken from AndorCamera.
"""

from __future__ import annotations

import os

os.environ["ANDOR_MOCK"] = "1"

from unittest.mock import MagicMock, patch

import pytest
from PySide6.QtWidgets import QApplication

READOUT_KEYS = {"amplifier_type_applied", "readout_flip_corrected", "readout_orientation"}


@pytest.fixture
def hw_manager(qt_app, hardware_manager, wait_for):
    completed = []
    hardware_manager.initialize(on_complete=lambda: completed.append(True))
    wait_for(lambda: len(completed) > 0)
    return hardware_manager


class TestMainWindowRecordsReadout:
    def test_button_params_carry_applied_readout(self, hw_manager, wait_for):
        from andor_qt.windows.main_window import AndorSpectrometerWindow

        window = AndorSpectrometerWindow()
        params = {
            "exposure_time": 0.01, "hbin": 1, "read_mode": "fvb", "num_accumulations": 1,
            "camera_settings": {"amplifier_type": 1, "trigger_mode": "internal",
                                "exposure_time": 0.01},
        }
        got = []
        window._acq_signals.spectrum_ready.connect(lambda cal, data, p: got.append(p))
        with patch.object(window._inputs_widget, "get_parameters", return_value=params), \
                patch.object(window, "_save_data"):
            window._on_queue_clicked()
            # spectrum_ready is emitted from the worker thread and queued to
            # this (main) thread, so Qt events must be processed while waiting
            assert wait_for(lambda: QApplication.processEvents() or len(got) > 0)

        assert READOUT_KEYS <= set(got[0])
        assert got[0]["amplifier_type_applied"] == 1
        assert got[0]["readout_flip_corrected"] == 0
        assert got[0]["readout_orientation"] == "short_wavelength_first"


class TestQueueRecordsReadout:
    def test_queue_params_carry_applied_readout(self, hw_manager):
        from andor_qt.core.experiment_queue import ExperimentQueueRunner
        from andor_qt.procedures import SpectrumProcedure

        proc = SpectrumProcedure()
        proc.exposure_time = 0.01
        proc.center_wavelength = 500.0
        proc.grating = 1
        proc.hbin = 1
        proc.num_accumulations = 1
        proc.amplifier_type = 0
        proc.em_gain = 1

        runner = ExperimentQueueRunner(hw_manager)
        got = []
        runner.spectrum_ready.connect(lambda cal, data, p: got.append(p))
        runner._execute_procedure(proc)

        assert got and READOUT_KEYS <= set(got[0])
        assert got[0]["amplifier_type_applied"] == 0
        assert got[0]["readout_flip_corrected"] == 1


@pytest.fixture
def ta_hw_manager():
    mgr = MagicMock()
    mgr.motion_manager.get_axis.return_value = None
    mgr.motion_manager.all_axes = {}
    mgr.get_wavelengths.return_value = [500.0, 600.0, 700.0]
    mgr.camera.readout_metadata.return_value = {
        "amplifier_type_applied": 1,
        "readout_flip_corrected": 0,
        "readout_orientation": "short_wavelength_first",
    }
    return mgr


class TestTAPanelRecordsReadout:
    def test_finalize_writes_readout_to_h5_metadata(self, qt_app, ta_hw_manager):
        from andor_qt.windows.ta_panel import TAWindowPanel

        panel = TAWindowPanel(hw_manager=ta_hw_manager)
        writer = MagicMock()
        panel._writer = writer
        panel._finalize_writer()

        writer.update_metadata.assert_called_once_with(
            ta_hw_manager.camera.readout_metadata.return_value
        )
        writer.finalize.assert_called_once()
        panel.deleteLater()

    def test_end_scan_stores_readout_for_post_scan_save(self, qt_app, ta_hw_manager):
        from andor_qt.windows.ta_panel import TAWindowPanel

        panel = TAWindowPanel(hw_manager=ta_hw_manager)
        panel._reset_scan_buffer(MagicMock(), {"amplifier_type": 1})
        panel._end_scan()

        assert panel._last_scan_buffer["readout"] == \
            ta_hw_manager.camera.readout_metadata.return_value
        panel.deleteLater()

    def test_non_dict_readout_is_ignored(self, qt_app, ta_hw_manager):
        """A camera without readout_metadata (e.g. a bare MagicMock) writes nothing."""
        from andor_qt.windows.ta_panel import TAWindowPanel

        ta_hw_manager.camera.readout_metadata.return_value = MagicMock()
        panel = TAWindowPanel(hw_manager=ta_hw_manager)
        writer = MagicMock()
        panel._writer = writer
        panel._finalize_writer()

        writer.update_metadata.assert_not_called()
        writer.finalize.assert_called_once()
        panel.deleteLater()
