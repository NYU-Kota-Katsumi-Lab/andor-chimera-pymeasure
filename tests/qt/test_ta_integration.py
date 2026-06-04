"""Tests for TA module integration into main window.

Verifies that TAWindowPanel is created and wired correctly,
and that the main window has a TA tab.
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

os.environ["ANDOR_MOCK"] = "1"


@pytest.fixture
def mock_hw_manager():
    """Create a minimal mock HardwareManager."""
    mgr = MagicMock()
    mgr.camera = MagicMock()
    mgr.spectrograph = MagicMock()
    mgr.motion_manager = MagicMock()
    mgr.motion_manager.get_axis.return_value = None
    mgr.motion_manager.all_axes = {}
    mgr.get_wavelengths.return_value = [500.0, 600.0, 700.0]
    return mgr


class TestTAWindowPanel:
    def test_creates_successfully(self, qt_app, mock_hw_manager):
        from andor_qt.windows.ta_panel import TAWindowPanel
        panel = TAWindowPanel(hw_manager=mock_hw_manager)
        assert panel is not None
        panel.deleteLater()

    def test_has_config_widget(self, qt_app, mock_hw_manager):
        from andor_qt.windows.ta_panel import TAWindowPanel
        panel = TAWindowPanel(hw_manager=mock_hw_manager)
        assert panel.config_widget is not None
        panel.deleteLater()

    def test_has_live_display(self, qt_app, mock_hw_manager):
        from andor_qt.windows.ta_panel import TAWindowPanel
        panel = TAWindowPanel(hw_manager=mock_hw_manager)
        assert panel.live_display is not None
        panel.deleteLater()

    def test_has_engine(self, qt_app, mock_hw_manager):
        from andor_qt.windows.ta_panel import TAWindowPanel
        from andor_qt.ta.engine import TransientAbsorptionEngine
        panel = TAWindowPanel(hw_manager=mock_hw_manager)
        assert isinstance(panel.engine, TransientAbsorptionEngine)
        panel.deleteLater()

    def test_scan_requested_connects_to_engine(self, qt_app, mock_hw_manager):
        """scan_requested signal from config widget should trigger engine start."""
        from andor_qt.windows.ta_panel import TAWindowPanel
        panel = TAWindowPanel(hw_manager=mock_hw_manager)
        # Should not raise when scan is requested with mock hw
        from andor_qt.ta.scan_config import TAScanConfig
        config = TAScanConfig(
            delay_list=[0.0], n_averages=1, n_scans=1,
            acquisition_mode="boxcar", scan_direction="forward",
            sample_name="test",
        )
        # Just verify emitting doesn't crash
        panel.config_widget.scan_requested.emit(config)
        panel.engine.abort()
        panel.deleteLater()

    def test_has_chopper_widget_when_manager_has_pair(self, qt_app, mock_hw_manager):
        """TAWindowPanel exposes a ChopperControlWidget bound to hw.chopper_pair."""
        from andor_pymeasure.instruments.thorlabs_mc2000b import (
            ChopperPair,
            MockMC2000B,
        )
        from andor_qt.widgets.hardware.chopper_control import ChopperControlWidget
        from andor_qt.windows.ta_panel import TAWindowPanel

        mock_hw_manager.chopper_pair = ChopperPair(
            pump=MockMC2000B(port="COM_TEST_PUMP"),
            probe=MockMC2000B(port="COM_TEST_PROBE"),
        )
        panel = TAWindowPanel(hw_manager=mock_hw_manager)
        assert isinstance(panel.chopper_widget, ChopperControlWidget)
        assert panel.chopper_widget.setup_button.isEnabled() is True
        panel.deleteLater()

    def test_chopper_widget_disabled_when_no_pair(self, qt_app, mock_hw_manager):
        """When HardwareManager has no chopper_pair, widget is constructed but disabled."""
        from andor_qt.windows.ta_panel import TAWindowPanel

        mock_hw_manager.chopper_pair = None
        panel = TAWindowPanel(hw_manager=mock_hw_manager)
        assert panel.chopper_widget.setup_button.isEnabled() is False
        panel.deleteLater()

    def test_camera_busy_disables_chopper_writes(self, qt_app, mock_hw_manager):
        """Emitting camera_busy(True) should disable the chopper widget's writes."""
        from andor_pymeasure.instruments.thorlabs_mc2000b import (
            ChopperPair,
            MockMC2000B,
        )
        from andor_qt.windows.ta_panel import TAWindowPanel

        mock_hw_manager.chopper_pair = ChopperPair(
            pump=MockMC2000B(port="COM_TEST_PUMP"),
            probe=MockMC2000B(port="COM_TEST_PROBE"),
        )
        panel = TAWindowPanel(hw_manager=mock_hw_manager)
        assert panel.chopper_widget.setup_button.isEnabled() is True

        panel.camera_busy.emit(True)
        assert panel.chopper_widget.setup_button.isEnabled() is False

        panel.camera_busy.emit(False)
        assert panel.chopper_widget.setup_button.isEnabled() is True
        panel.deleteLater()

    def test_chopper_widget_rebinds_after_late_init(self, qt_app, mock_hw_manager):
        """Panel built before chopper_pair exists; widget rebinds on signal."""
        from andor_pymeasure.instruments.thorlabs_mc2000b import (
            ChopperPair,
            MockMC2000B,
        )
        from andor_qt.core.signals import get_hardware_signals
        from andor_qt.windows.ta_panel import TAWindowPanel

        # Construct the panel BEFORE the hardware manager has a chopper_pair.
        mock_hw_manager.chopper_pair = None
        panel = TAWindowPanel(hw_manager=mock_hw_manager)
        assert panel.chopper_widget.setup_button.isEnabled() is False

        # Now simulate the hardware-manager finishing chopper init:
        mock_hw_manager.chopper_pair = ChopperPair(
            pump=MockMC2000B(port="COM_LATE_PUMP"),
            probe=MockMC2000B(port="COM_LATE_PROBE"),
        )
        get_hardware_signals().choppers_initialized.emit(
            {"pump_port": "COM_LATE_PUMP", "probe_port": "COM_LATE_PROBE",
             "pump_connected": True, "probe_connected": True}
        )
        qt_app.processEvents()

        # Widget should now be live and bound to the new pair.
        assert panel.chopper_widget.setup_button.isEnabled() is True
        assert "COM_LATE_PUMP" in panel.chopper_widget.pump_panel._port_label.text()
        assert "COM_LATE_PROBE" in panel.chopper_widget.probe_panel._port_label.text()
        panel.deleteLater()


class TestChopper2x2Panel:
    """TAWindowPanel creates mock NI DAQ objects for chopper_2x2 mode."""

    def _make_panel(self, mock_hw_manager, qt_app):
        from andor_qt.windows.ta_panel import TAWindowPanel
        return TAWindowPanel(hw_manager=mock_hw_manager)

    def test_chopper_2x2_passes_trigger_gen_to_engine(self, qt_app, mock_hw_manager):
        from unittest.mock import patch
        from andor_qt.ta.scan_config import TAScanConfig
        from andor_qt.windows.ta_panel import TAWindowPanel

        panel = TAWindowPanel(hw_manager=mock_hw_manager)
        config = TAScanConfig(
            delay_list=[0.0], n_averages=1, n_scans=1,
            acquisition_mode="chopper_2x2", scan_direction="forward",
            sample_name="test",
        )
        captured = {}
        original = panel._engine.start_scan
        def _capture(*args, **kwargs):
            captured.update(kwargs)
            return original(*args, **kwargs)

        with patch.object(panel._engine, "start_scan", side_effect=_capture):
            panel.config_widget.scan_requested.emit(config)

        panel.engine.abort()
        panel.deleteLater()
        assert captured.get("trigger_gen") is not None

    def test_chopper_2x2_passes_phase_reader_to_engine(self, qt_app, mock_hw_manager):
        from unittest.mock import patch
        from andor_qt.ta.scan_config import TAScanConfig
        from andor_qt.windows.ta_panel import TAWindowPanel

        panel = TAWindowPanel(hw_manager=mock_hw_manager)
        config = TAScanConfig(
            delay_list=[0.0], n_averages=1, n_scans=1,
            acquisition_mode="chopper_2x2", scan_direction="forward",
            sample_name="test",
        )
        captured = {}
        original = panel._engine.start_scan
        def _capture(*args, **kwargs):
            captured.update(kwargs)
            return original(*args, **kwargs)

        with patch.object(panel._engine, "start_scan", side_effect=_capture):
            panel.config_widget.scan_requested.emit(config)

        panel.engine.abort()
        panel.deleteLater()
        assert captured.get("phase_reader") is not None

    def test_non_chopper_mode_has_no_trigger_gen(self, qt_app, mock_hw_manager):
        from unittest.mock import patch
        from andor_qt.ta.scan_config import TAScanConfig
        from andor_qt.windows.ta_panel import TAWindowPanel

        panel = TAWindowPanel(hw_manager=mock_hw_manager)
        config = TAScanConfig(
            delay_list=[0.0], n_averages=1, n_scans=1,
            acquisition_mode="boxcar", scan_direction="forward",
            sample_name="test",
        )
        captured = {}
        original = panel._engine.start_scan
        def _capture(*args, **kwargs):
            captured.update(kwargs)
            return original(*args, **kwargs)

        with patch.object(panel._engine, "start_scan", side_effect=_capture):
            panel.config_widget.scan_requested.emit(config)

        panel.engine.abort()
        panel.deleteLater()
        assert captured.get("trigger_gen") is None
        assert captured.get("phase_reader") is None

    def test_chopper_2x2_uses_mock_objects_in_mock_mode(self, qt_app, mock_hw_manager):
        from andor_qt.ta.scan_config import TAScanConfig
        from andor_qt.ta.nidaq_trigger import MockNIDAQChopper500Hz
        from andor_qt.ta.nidaq_phase import MockNIDAQChopper2x2Reader
        from andor_qt.windows.ta_panel import TAWindowPanel

        panel = TAWindowPanel(hw_manager=mock_hw_manager)
        config = TAScanConfig(
            delay_list=[0.0], n_averages=1, n_scans=1,
            acquisition_mode="chopper_2x2", scan_direction="forward",
            sample_name="test",
        )
        captured = {}
        original = panel._engine.start_scan
        def _capture(*args, **kwargs):
            captured.update(kwargs)
            return original(*args, **kwargs)

        with patch.object(panel._engine, "start_scan", side_effect=_capture):
            panel.config_widget.scan_requested.emit(config)

        panel.engine.abort()
        panel.deleteLater()
        assert isinstance(captured.get("trigger_gen"), MockNIDAQChopper500Hz)
        from andor_qt.ta.nidaq_phase import MockNIDAQPhaseReader
        assert isinstance(captured.get("phase_reader"), MockNIDAQPhaseReader)


class TestShotToShotPanel:
    def test_shot_to_shot_creates_phase_reader_only(self, qt_app, mock_hw_manager):
        """shot_to_shot mode should create a phase reader but no trigger gen."""
        from unittest.mock import patch
        from andor_qt.windows.ta_panel import TAWindowPanel
        from andor_qt.ta.scan_config import TAScanConfig

        panel = TAWindowPanel(hw_manager=mock_hw_manager)
        config = TAScanConfig(
            delay_list=[0.0],
            n_averages=1,
            n_scans=1,
            acquisition_mode="shot_to_shot",
            scan_direction="forward",
            sample_name="test",
            crop_height=50,
        )

        captured = {}
        def _capture(*args, **kwargs):
            captured.update(kwargs)
        with patch.object(panel._engine, "start_scan", side_effect=_capture):
            panel.config_widget.scan_requested.emit(config)

        panel.engine.abort()
        panel.deleteLater()
        assert captured.get("trigger_gen") is None
        assert captured.get("phase_reader") is not None


class TestCameraBusySignal:
    """TAWindowPanel.camera_busy should fire True/False around scan and monitor."""

    def test_camera_busy_signal_exists(self, qt_app, mock_hw_manager):
        from andor_qt.windows.ta_panel import TAWindowPanel
        panel = TAWindowPanel(hw_manager=mock_hw_manager)
        assert hasattr(panel, "camera_busy"), "TAWindowPanel must have camera_busy signal"
        panel.deleteLater()

    def test_camera_busy_true_on_scan_start(self, qt_app, mock_hw_manager):
        """camera_busy(True) should be emitted when a scan starts."""
        from andor_qt.windows.ta_panel import TAWindowPanel
        from andor_qt.ta.scan_config import TAScanConfig

        panel = TAWindowPanel(hw_manager=mock_hw_manager)
        received = []
        panel.camera_busy.connect(lambda v: received.append(v))

        config = TAScanConfig(
            delay_list=[0.0], n_averages=1, n_scans=1,
            acquisition_mode="boxcar", scan_direction="forward",
            sample_name="test",
        )
        panel.config_widget.scan_requested.emit(config)
        panel.engine.abort()
        panel.deleteLater()
        assert True in received, "camera_busy(True) not emitted on scan start"

    def test_camera_busy_false_on_scan_end(self, qt_app, mock_hw_manager):
        """camera_busy(False) should be emitted when a scan completes/aborts."""
        from andor_qt.windows.ta_panel import TAWindowPanel
        from andor_qt.ta.scan_config import TAScanConfig
        from PySide6.QtWidgets import QApplication
        import time

        panel = TAWindowPanel(hw_manager=mock_hw_manager)
        received = []
        panel.camera_busy.connect(lambda v: received.append(v))

        config = TAScanConfig(
            delay_list=[0.0], n_averages=1, n_scans=1,
            acquisition_mode="boxcar", scan_direction="forward",
            sample_name="test",
        )
        panel.config_widget.scan_requested.emit(config)
        panel.engine.abort()

        # Process events to let signals propagate
        app = QApplication.instance()
        deadline = time.time() + 5.0
        while time.time() < deadline:
            app.processEvents()
            if False in received:
                break
            time.sleep(0.05)

        panel.deleteLater()
        assert False in received, "camera_busy(False) not emitted on scan end"

    def test_camera_busy_true_on_monitor_start(self, qt_app, mock_hw_manager):
        """camera_busy(True) should be emitted when monitor starts."""
        from andor_qt.windows.ta_panel import TAWindowPanel
        from andor_qt.ta.scan_config import TAScanConfig

        panel = TAWindowPanel(hw_manager=mock_hw_manager)
        received = []
        panel.camera_busy.connect(lambda v: received.append(v))

        config = TAScanConfig(
            delay_list=[0.0], n_averages=1,
            acquisition_mode="boxcar", scan_direction="forward",
            sample_name="monitor",
        )
        panel._on_monitor_requested(config)
        panel._monitor_engine.stop()

        panel.deleteLater()
        assert True in received, "camera_busy(True) not emitted on monitor start"

    def test_camera_busy_false_on_monitor_stop(self, qt_app, mock_hw_manager):
        """camera_busy(False) should be emitted when monitor stops."""
        from andor_qt.windows.ta_panel import TAWindowPanel
        from andor_qt.ta.scan_config import TAScanConfig
        from PySide6.QtWidgets import QApplication
        import time

        panel = TAWindowPanel(hw_manager=mock_hw_manager)
        received = []
        panel.camera_busy.connect(lambda v: received.append(v))

        config = TAScanConfig(
            delay_list=[0.0], n_averages=1,
            acquisition_mode="boxcar", scan_direction="forward",
            sample_name="monitor",
        )
        panel._on_monitor_requested(config)
        panel._monitor_engine.stop()

        app = QApplication.instance()
        deadline = time.time() + 5.0
        while time.time() < deadline:
            app.processEvents()
            if False in received:
                break
            time.sleep(0.05)

        panel.deleteLater()
        assert False in received, "camera_busy(False) not emitted on monitor stop"


class TestMainWindowAcquireLockout:
    """Main window acquire/queue buttons should be disabled during TA camera use."""

    def test_acquire_disabled_during_ta_scan(self, qt_app, mock_sdk, reset_hardware_manager):
        from andor_qt.windows.main_window import AndorSpectrometerWindow

        window = AndorSpectrometerWindow()
        # Simulate TA camera busy
        window._ta_panel.camera_busy.emit(True)

        assert not window._acquire_control._acquire_btn.isEnabled(), \
            "Acquire button should be disabled when TA camera is busy"
        window.deleteLater()

    def test_acquire_reenabled_after_ta_scan(self, qt_app, mock_sdk, reset_hardware_manager):
        from andor_qt.windows.main_window import AndorSpectrometerWindow

        window = AndorSpectrometerWindow()
        window._ta_panel.camera_busy.emit(True)
        window._ta_panel.camera_busy.emit(False)

        assert window._acquire_control._acquire_btn.isEnabled(), \
            "Acquire button should be re-enabled when TA camera is free"
        window.deleteLater()

    def test_queue_disabled_during_ta_scan(self, qt_app, mock_sdk, reset_hardware_manager):
        from andor_qt.windows.main_window import AndorSpectrometerWindow

        window = AndorSpectrometerWindow()
        window._ta_panel.camera_busy.emit(True)

        assert not window._queue_control._queue_button.isEnabled(), \
            "Queue button should be disabled when TA camera is busy"
        window.deleteLater()


class TestMainWindowTATab:
    def test_main_window_has_ta_tab(self, qt_app, mock_sdk, reset_hardware_manager):
        """Main window should have a TA tab."""
        from unittest.mock import patch
        from PySide6.QtWidgets import QTabWidget
        from andor_qt.windows.main_window import AndorSpectrometerWindow

        window = AndorSpectrometerWindow()
        # Find top-level tab widget
        tabs = window.findChildren(QTabWidget)
        assert len(tabs) > 0

        # One of them should have a tab called "TA" or "Transient Absorption"
        ta_tab_found = False
        for tab_widget in tabs:
            for i in range(tab_widget.count()):
                label = tab_widget.tabText(i)
                if "TA" in label or "Transient" in label:
                    ta_tab_found = True
                    break

        # Skip close() to avoid triggering the blocking shutdown sequence in tests
        window.deleteLater()

        assert ta_tab_found, "No TA tab found in main window"


class TestHbinWavelengthAlignment:
    """Wavelength arrays must match hbin-reduced pixel count."""

    def test_hdf5_writer_gets_hbin_adjusted_wavelengths(self, qt_app, mock_hw_manager, tmp_path):
        """HDF5 writer wavelength array should have xpixels/hbin points."""
        import numpy as np
        from andor_qt.windows.ta_panel import TAWindowPanel
        from andor_qt.ta.scan_config import TAScanConfig

        # Mock get_wavelengths to return different lengths based on hbin
        def _mock_get_wl(hbin=1):
            n_pixels = 1600 // hbin
            return np.linspace(400, 800, n_pixels)
        mock_hw_manager.get_wavelengths.side_effect = _mock_get_wl

        panel = TAWindowPanel(hw_manager=mock_hw_manager)
        config = TAScanConfig(
            delay_list=[0.0], n_averages=1, n_scans=1,
            acquisition_mode="boxcar", scan_direction="forward",
            sample_name="test",
            save_hdf5_dir=str(tmp_path),
        )

        # Set hbin=8 in camera settings
        panel._config_widget._camera_settings.hbin_combo.setCurrentIndex(3)  # 8x
        captured = {}
        def _capture(*args, **kwargs):
            captured.update(kwargs)
        with patch.object(panel._engine, "start_scan", side_effect=_capture):
            panel.config_widget.scan_requested.emit(config)

        panel.engine.abort()

        # Verify get_wavelengths was called with hbin=8
        mock_hw_manager.get_wavelengths.assert_called_with(hbin=8)

        # Verify the writer was created with 200 wavelength points
        writer = panel._writer
        if writer is not None:
            assert len(writer._wavelengths) == 200, (
                f"Expected 200 wavelengths (1600/8), got {len(writer._wavelengths)}"
            )
            writer.finalize()
        panel.deleteLater()

    def test_scan_engine_gets_hbin_adjusted_wavelengths(self, qt_app, mock_hw_manager):
        """Scan engine should pass hbin to get_wavelengths via ta_panel."""
        import numpy as np
        from andor_qt.windows.ta_panel import TAWindowPanel
        from andor_qt.ta.scan_config import TAScanConfig

        def _mock_get_wl(hbin=1):
            return np.linspace(400, 800, 1600 // hbin)
        mock_hw_manager.get_wavelengths.side_effect = _mock_get_wl

        panel = TAWindowPanel(hw_manager=mock_hw_manager)
        # Set hbin=4 in camera settings
        panel._config_widget._camera_settings.hbin_combo.setCurrentIndex(2)  # 4x

        config = TAScanConfig(
            delay_list=[0.0], n_averages=1, n_scans=1,
            acquisition_mode="boxcar", scan_direction="forward",
            sample_name="test",
        )
        captured = {}
        def _capture(*args, **kwargs):
            captured.update(kwargs)
        with patch.object(panel._engine, "start_scan", side_effect=_capture):
            panel.config_widget.scan_requested.emit(config)

        panel.engine.abort()
        panel.deleteLater()
        # camera_settings should have hbin=4
        cs = captured.get("camera_settings", {})
        assert cs.get("hbin") == 4

    def test_monitor_engine_gets_hbin_adjusted_wavelengths(self, qt_app, mock_hw_manager):
        """Monitor pre_set_wavelengths should pass hbin to get_wavelengths."""
        import numpy as np
        from andor_qt.windows.ta_panel import TAWindowPanel

        def _mock_get_wl(hbin=1):
            return np.linspace(400, 800, 1600 // hbin)
        mock_hw_manager.get_wavelengths.side_effect = _mock_get_wl

        panel = TAWindowPanel(hw_manager=mock_hw_manager)
        panel._pre_set_wavelengths({"hbin": 4})

        mock_hw_manager.get_wavelengths.assert_called_with(hbin=4)
        assert len(panel._live_display._wavelengths) == 400
        panel.deleteLater()

    def test_pre_set_wavelengths_passes_hbin(self, qt_app, mock_hw_manager):
        """_pre_set_wavelengths should pass hbin from camera_settings."""
        import numpy as np
        from andor_qt.windows.ta_panel import TAWindowPanel

        def _mock_get_wl(hbin=1):
            return np.linspace(400, 800, 1600 // hbin)
        mock_hw_manager.get_wavelengths.side_effect = _mock_get_wl

        panel = TAWindowPanel(hw_manager=mock_hw_manager)
        panel._pre_set_wavelengths({"hbin": 8})

        mock_hw_manager.get_wavelengths.assert_called_with(hbin=8)
        assert len(panel._live_display._wavelengths) == 200
        panel.deleteLater()


class TestStageAxisE2E:
    """End-to-end: spinbox value in config widget → engine → set_axis_hardware_index."""

    def test_spinbox_value_in_emitted_config(self, qt_app, mock_hw_manager):
        """Config emitted by scan_requested carries the spinbox stage_axis value."""
        from andor_qt.windows.ta_panel import TAWindowPanel

        panel = TAWindowPanel(hw_manager=mock_hw_manager)
        # Change the spinbox to axis 3
        panel.config_widget._stage_axis_spin.setValue(3)

        emitted_configs = []
        panel.config_widget.scan_requested.connect(lambda cfg: emitted_configs.append(cfg))
        panel.config_widget.scan_requested.emit(panel.config_widget._build_config())

        assert len(emitted_configs) == 1
        assert emitted_configs[0].stage_axis == 3
        panel.deleteLater()

    def test_spinbox_axis_1_in_emitted_config(self, qt_app, mock_hw_manager):
        """Config emitted by scan_requested carries stage_axis=1."""
        from andor_qt.windows.ta_panel import TAWindowPanel

        panel = TAWindowPanel(hw_manager=mock_hw_manager)
        panel.config_widget._stage_axis_spin.setValue(1)

        emitted_configs = []
        panel.config_widget.scan_requested.connect(lambda cfg: emitted_configs.append(cfg))
        panel.config_widget.scan_requested.emit(panel.config_widget._build_config())

        assert len(emitted_configs) == 1
        assert emitted_configs[0].stage_axis == 1
        panel.deleteLater()

    def test_engine_receives_stage_axis_from_panel(self, qt_app, mock_hw_manager):
        """Full flow: spinbox=3 → panel starts engine → engine calls set_axis_hardware_index(3)."""
        import time
        from PySide6.QtWidgets import QApplication
        from andor_qt.windows.ta_panel import TAWindowPanel

        panel = TAWindowPanel(hw_manager=mock_hw_manager)
        panel.config_widget._stage_axis_spin.setValue(3)

        # Simulate a scan request
        config = panel.config_widget._build_config()
        config.delay_list = [0.0]
        config.n_averages = 1
        config.n_scans = 1

        done = [False]
        panel.engine.scan_completed.connect(lambda: done.__setitem__(0, True))
        panel.engine.aborted.connect(lambda: done.__setitem__(0, True))
        panel.engine.error.connect(lambda _: done.__setitem__(0, True))

        panel.engine.start_scan(config, mock_hw_manager)

        start = time.time()
        while not done[0] and time.time() - start < 10.0:
            QApplication.instance().processEvents()
            time.sleep(0.005)

        mock_hw_manager.motion_manager.set_axis_hardware_index.assert_called_with("delay", 3)
        panel.deleteLater()


class TestPostScanSave:
    """Post-scan save buffer captures data for later HDF5 export."""

    def test_buffer_starts_empty(self, qt_app, mock_hw_manager):
        from andor_qt.windows.ta_panel import TAWindowPanel
        panel = TAWindowPanel(hw_manager=mock_hw_manager)
        assert panel._last_scan_buffer["delays_ps"] == []
        assert panel._last_scan_buffer["delta_signals"] == []
        assert panel._last_scan_buffer["wavelengths"] is None
        assert panel._save_last_btn.isEnabled() is False
        panel.deleteLater()

    def test_signal_updated_appends_to_buffer(self, qt_app, mock_hw_manager):
        """signal_updated handler appends delay, delta, and correctly computed stage position."""
        import numpy as np
        from andor_qt.windows.ta_panel import TAWindowPanel
        from andor_pymeasure.instruments.motion_controller import SPEED_OF_LIGHT_MM_PS
        panel = TAWindowPanel(hw_manager=mock_hw_manager)

        wl = np.linspace(400.0, 800.0, 5)
        delta = np.array([0.01, 0.02, 0.03, 0.02, 0.01])
        # 1.5 ps → stage position = 1.5 * c_mm_per_ps / 2 * 1000 µm
        expected_pos_um = (1.5 * SPEED_OF_LIGHT_MM_PS / 2.0) * 1000.0
        panel._on_buffer_signal_updated(1.5, wl, delta)

        buf = panel._last_scan_buffer
        assert buf["delays_ps"] == [1.5]
        assert len(buf["delta_signals"]) == 1
        np.testing.assert_array_equal(buf["delta_signals"][0], delta)
        np.testing.assert_array_equal(buf["wavelengths"], wl)
        assert len(buf["stage_positions_um"]) == 1
        # Verify stage position is computed correctly (not just "exists")
        assert buf["stage_positions_um"][0] == pytest.approx(expected_pos_um)
        # Hand-calc sanity: 1.5 ps → ~224.84 µm
        assert buf["stage_positions_um"][0] == pytest.approx(224.8443435, rel=1e-6)
        panel.deleteLater()

    def test_multiple_delays_produce_correct_stage_positions(self, qt_app, mock_hw_manager):
        """Each signal_updated call produces the correct stage position for its delay."""
        import numpy as np
        from andor_qt.windows.ta_panel import TAWindowPanel
        from andor_pymeasure.instruments.motion_controller import SPEED_OF_LIGHT_MM_PS
        panel = TAWindowPanel(hw_manager=mock_hw_manager)

        wl = np.linspace(400.0, 800.0, 3)
        delays = [-2.0, 0.0, 5.0, 10.0]
        for d in delays:
            panel._on_buffer_signal_updated(d, wl, np.zeros(3))

        buf = panel._last_scan_buffer
        assert buf["delays_ps"] == delays
        for d, pos in zip(delays, buf["stage_positions_um"]):
            expected = (d * SPEED_OF_LIGHT_MM_PS / 2.0) * 1000.0
            assert pos == pytest.approx(expected)
        panel.deleteLater()

    def test_raw_pair_updated_appends_pump_ref(self, qt_app, mock_hw_manager):
        """raw_pair_updated handler appends pump and ref spectra."""
        import numpy as np
        from andor_qt.windows.ta_panel import TAWindowPanel
        panel = TAWindowPanel(hw_manager=mock_hw_manager)

        pump = np.ones(5) * 1200.0
        ref = np.ones(5) * 1000.0
        panel._on_buffer_raw_pair_updated(pump, ref, 100, 0, 200)

        buf = panel._last_scan_buffer
        assert len(buf["pump_spectra"]) == 1
        assert len(buf["ref_spectra"]) == 1
        np.testing.assert_array_equal(buf["pump_spectra"][0], pump)
        np.testing.assert_array_equal(buf["ref_spectra"][0], ref)
        panel.deleteLater()

    def test_save_writes_hdf5(self, qt_app, mock_hw_manager, tmp_path):
        """_on_save_last_scan writes buffered data to HDF5 with correct values and metadata."""
        import numpy as np
        import h5py
        from unittest.mock import patch
        from andor_qt.ta.scan_config import TAScanConfig
        from andor_qt.windows.ta_panel import TAWindowPanel
        from andor_pymeasure.instruments.motion_controller import SPEED_OF_LIGHT_MM_PS

        panel = TAWindowPanel(hw_manager=mock_hw_manager)

        # Manually populate buffer with distinguishable delta values per point
        config = TAScanConfig(
            delay_list=[0.0, 1.0], n_averages=1, sample_name="test_sample",
            notes="post-save test", stage_axis=2,
        )
        panel._reset_scan_buffer(config, {"hbin": 1, "exposure_time": 0.001})
        wl = np.linspace(400.0, 800.0, 5)
        delays = [0.0, 1.0]
        # Each point has a unique delta profile (0.01..0.05, 0.02..0.06)
        expected_deltas = [
            np.array([0.01, 0.02, 0.03, 0.04, 0.05]),
            np.array([0.02, 0.03, 0.04, 0.05, 0.06]),
        ]
        for d, delta in zip(delays, expected_deltas):
            panel._on_buffer_signal_updated(d, wl, delta)
            panel._on_buffer_raw_pair_updated(
                np.ones(5) * 1200, np.ones(5) * 1000, 100, 0, 200
            )

        out_path = tmp_path / "post_save.h5"
        with patch(
            "PySide6.QtWidgets.QFileDialog.getSaveFileName",
            return_value=(str(out_path), ""),
        ), patch("PySide6.QtWidgets.QMessageBox.information"):
            panel._on_save_last_scan()

        assert out_path.exists()
        with h5py.File(out_path, "r") as f:
            # Structure
            assert "scan_000" in f
            assert "metadata" in f
            assert "wavelengths" in f

            # Time delays and delta values round-trip correctly
            np.testing.assert_allclose(f["scan_000/time_delays"][:], delays)
            for i, expected in enumerate(expected_deltas):
                np.testing.assert_allclose(f["scan_000/delta_signal"][i], expected)

            # Wavelength axis
            np.testing.assert_allclose(f["wavelengths"][:], wl)

            # Pump and ref spectra
            np.testing.assert_allclose(f["scan_000/pump_spectrum"][0], np.ones(5) * 1200)
            np.testing.assert_allclose(f["scan_000/pump_spectrum"][1], np.ones(5) * 1200)
            np.testing.assert_allclose(f["scan_000/ref_spectrum"][0], np.ones(5) * 1000)

            # Stage positions computed from delays via speed of light
            pos_ds = f["scan_000/stage_positions_um"][:]
            for i, d in enumerate(delays):
                assert pos_ds[i] == pytest.approx((d * SPEED_OF_LIGHT_MM_PS / 2.0) * 1000.0)

            # Metadata from config and camera settings
            meta = f["metadata"]
            assert meta.attrs["sample_name"] == "test_sample"
            assert meta.attrs["notes"] == "post-save test"
            assert meta.attrs["exposure_time_s"] == pytest.approx(0.001)
            assert meta.attrs["hbin"] == 1
        panel.deleteLater()

    def test_save_writes_hardware_group(self, qt_app, mock_hw_manager, tmp_path):
        """Post-scan HDF5 save includes a /hardware group with scan axis info."""
        import numpy as np
        import h5py
        from unittest.mock import patch
        from andor_qt.ta.scan_config import TAScanConfig
        from andor_qt.windows.ta_panel import TAWindowPanel

        # Provide a fully-numeric mock axis so TAMonitorWidget._update_position
        # can format positions as floats without error.
        mock_axis = MagicMock()
        mock_axis.index = 2
        mock_axis.t0_offset_mm = 1.23
        mock_axis.position = 0.0
        mock_axis.position_ps = 0.0
        mock_axis.position_min = 0.0
        mock_axis.position_max = 300.0
        mock_hw_manager.motion_manager.get_axis.return_value = mock_axis
        mock_hw_manager.motion_manager.all_axes = {"delay": mock_axis}
        # Spectrograph returns None attributes so the try/except path is exercised safely
        mock_hw_manager.spectrograph.info = None
        mock_hw_manager.spectrograph.grating = 1
        mock_hw_manager.spectrograph.wavelength = 550.0

        panel = TAWindowPanel(hw_manager=mock_hw_manager)
        config = TAScanConfig(delay_list=[0.0], n_averages=1, sample_name="hw_test", stage_axis=3)
        panel._reset_scan_buffer(config, {"hbin": 1, "exposure_time": 0.001})
        panel._on_buffer_signal_updated(0.0, np.linspace(400, 800, 3), np.zeros(3))
        panel._on_buffer_raw_pair_updated(np.ones(3) * 100, np.ones(3) * 90, 1, 0, 2)

        out_path = tmp_path / "hw.h5"
        with patch(
            "PySide6.QtWidgets.QFileDialog.getSaveFileName",
            return_value=(str(out_path), ""),
        ), patch("PySide6.QtWidgets.QMessageBox.information"):
            panel._on_save_last_scan()

        with h5py.File(out_path, "r") as f:
            assert "hardware" in f
            hw = f["hardware"]
            # stage_axis comes from config
            assert hw.attrs["stage_axis"] == 3
            # axis HW index comes from the mock axis object
            assert hw.attrs["stage_axis_hw_index"] == 2
            # center wavelength comes from spectrograph.wavelength
            assert hw.attrs["center_wavelength_nm"] == pytest.approx(550.0)
            assert hw.attrs["grating_index"] == 1
        panel.deleteLater()

    def test_save_records_em_gain_mode_and_actual(self, qt_app, mock_hw_manager, tmp_path):
        """/hardware records em_gain_mode and em_gain_actual so analysts can tell
        whether the saved em_gain integer is a physical factor (mode 3) or DAC."""
        import numpy as np
        import h5py
        from unittest.mock import patch
        from andor_qt.ta.scan_config import TAScanConfig
        from andor_qt.windows.ta_panel import TAWindowPanel

        # Real ints, not MagicMock attributes, so int() coercion is meaningful.
        mock_hw_manager.camera.em_gain_mode = 3
        mock_hw_manager.camera.get_em_gain = MagicMock(return_value=50)

        panel = TAWindowPanel(hw_manager=mock_hw_manager)
        config = TAScanConfig(delay_list=[0.0], n_averages=1, sample_name="em_test")
        panel._reset_scan_buffer(config, {"hbin": 1, "exposure_time": 0.001})
        panel._on_buffer_signal_updated(0.0, np.linspace(400, 800, 3), np.zeros(3))
        panel._on_buffer_raw_pair_updated(np.ones(3) * 100, np.ones(3) * 90, 1, 0, 2)

        out_path = tmp_path / "em_meta.h5"
        with patch(
            "PySide6.QtWidgets.QFileDialog.getSaveFileName",
            return_value=(str(out_path), ""),
        ), patch("PySide6.QtWidgets.QMessageBox.information"):
            panel._on_save_last_scan()

        with h5py.File(out_path, "r") as f:
            hw = f["hardware"]
            assert hw.attrs["em_gain_mode"] == 3
            assert hw.attrs["em_gain_actual"] == 50
        panel.deleteLater()

    def test_save_hdf5_with_mismatched_pump_ref_counts(self, qt_app, mock_hw_manager, tmp_path):
        """Save succeeds even when pump/ref spectra are shorter than delays."""
        import numpy as np
        import h5py
        from unittest.mock import patch
        from andor_qt.ta.scan_config import TAScanConfig
        from andor_qt.windows.ta_panel import TAWindowPanel

        panel = TAWindowPanel(hw_manager=mock_hw_manager)

        config = TAScanConfig(delay_list=[0.0, 1.0, 2.0], n_averages=1, sample_name="partial")
        panel._reset_scan_buffer(config, {"hbin": 1, "exposure_time": 0.001})
        wl = np.linspace(400.0, 800.0, 5)

        # 3 delta signals but only 2 pump/ref pairs (simulates late point failing
        # after signal_updated but before raw_pair_updated)
        for d in [0.0, 1.0, 2.0]:
            panel._on_buffer_signal_updated(d, wl, np.ones(5) * 0.01)
        for _ in range(2):
            panel._on_buffer_raw_pair_updated(
                np.ones(5) * 1200, np.ones(5) * 1000, 100, 0, 200
            )

        out_path = tmp_path / "partial.h5"
        with patch(
            "PySide6.QtWidgets.QFileDialog.getSaveFileName",
            return_value=(str(out_path), ""),
        ), patch("PySide6.QtWidgets.QMessageBox.information"):
            panel._on_save_last_scan()

        # The save should succeed: 3 delta points written, only 2 pump/ref points
        assert out_path.exists()
        with h5py.File(out_path, "r") as f:
            assert f["scan_000/time_delays"].shape == (3,)
            assert f["scan_000/delta_signal"].shape == (3, 5)
            # pump/ref were only written for points where both were available
            assert f["scan_000/pump_spectrum"].shape == (2, 5)
            assert f["scan_000/ref_spectrum"].shape == (2, 5)
        panel.deleteLater()

    def test_save_spectra_writes_text_files(self, qt_app, mock_hw_manager, tmp_path):
        """_on_save_last_spectra writes per-point text files to a timestamped subfolder."""
        import numpy as np
        from unittest.mock import patch
        from andor_qt.ta.scan_config import TAScanConfig
        from andor_qt.windows.ta_panel import TAWindowPanel

        panel = TAWindowPanel(hw_manager=mock_hw_manager)

        config = TAScanConfig(delay_list=[0.0, 1.0], n_averages=1, sample_name="test")
        panel._reset_scan_buffer(config, {"hbin": 1, "exposure_time": 0.001})
        wl = np.linspace(400.0, 800.0, 5)
        for i, d in enumerate([0.0, 1.0]):
            panel._on_buffer_signal_updated(d, wl, np.ones(5) * (0.01 * (i + 1)))
            panel._on_buffer_raw_pair_updated(
                np.ones(5) * 1200, np.ones(5) * 1000, 100, 0, 200
            )

        with patch(
            "PySide6.QtWidgets.QFileDialog.getExistingDirectory",
            return_value=str(tmp_path),
        ), patch("PySide6.QtWidgets.QMessageBox.information"):
            panel._on_save_last_spectra()

        # The save creates a timestamped subfolder under tmp_path
        subfolders = [p for p in tmp_path.iterdir() if p.is_dir()]
        assert len(subfolders) == 1
        folder = subfolders[0]

        delta_files = list(folder.glob("scan000_pos*um.txt"))
        # 2 delta files + matching _pump and _ref text files
        delta_only = [f for f in delta_files if "_pump" not in f.name and "_ref" not in f.name]
        pump_files = list(folder.glob("scan000_pos*um_pump.txt"))
        ref_files = list(folder.glob("scan000_pos*um_ref.txt"))
        assert len(delta_only) == 2
        assert len(pump_files) == 2
        assert len(ref_files) == 2

        # Verify content of one delta file: 5 tab-separated rows of wl<TAB>delta
        first = sorted(delta_only)[0]
        lines = first.read_text().strip().split("\n")
        assert len(lines) == 5
        for line, expected_wl in zip(lines, wl):
            cols = line.split("\t")
            assert len(cols) == 2
            assert float(cols[0]) == pytest.approx(expected_wl, rel=1e-3)
            # first delta file is for delay=0 → delta = 0.01
            assert float(cols[1]) == pytest.approx(0.01, rel=1e-6)

        # Verify pump text file content: 5 rows, all value 1200
        first_pump = sorted(pump_files)[0]
        pump_lines = first_pump.read_text().strip().split("\n")
        assert len(pump_lines) == 5
        for line in pump_lines:
            cols = line.split("\t")
            assert float(cols[1]) == pytest.approx(1200.0)
        panel.deleteLater()

    def test_save_spectra_no_data_shows_message(self, qt_app, mock_hw_manager):
        """Saving with no buffered data shows a message and doesn't crash."""
        from unittest.mock import patch
        from andor_qt.windows.ta_panel import TAWindowPanel

        panel = TAWindowPanel(hw_manager=mock_hw_manager)
        # buffer is empty by default
        with patch("PySide6.QtWidgets.QMessageBox.information") as mock_info:
            panel._on_save_last_spectra()
        mock_info.assert_called_once()
        panel.deleteLater()
