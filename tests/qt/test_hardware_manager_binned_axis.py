"""HardwareManager.get_calibration(hbin) must return bin-centred wavelengths."""

from __future__ import annotations

import os

os.environ["ANDOR_MOCK"] = "1"

import numpy as np


class TestHardwareManagerUsesBinCentres:
    def test_get_calibration_hbin8_is_bin_centred(self, qt_app, hardware_manager, wait_for):
        completed = []
        hardware_manager.initialize(on_complete=lambda: completed.append(True))
        wait_for(lambda: len(completed) > 0)

        cam = hardware_manager.camera
        unbinned = hardware_manager.get_calibration(hbin=1)
        binned = hardware_manager.get_calibration(hbin=8)
        assert binned.shape == (cam.xpixels // 8,)
        np.testing.assert_allclose(binned, unbinned.reshape(-1, 8).mean(axis=1))
