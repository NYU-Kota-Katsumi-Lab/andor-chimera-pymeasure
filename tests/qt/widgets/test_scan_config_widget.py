"""Tests for TAScanConfigWidget."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from PySide6.QtWidgets import QApplication

from andor_qt.widgets.ta.scan_config_widget import TAScanConfigWidget
from andor_qt.ta.scan_config import TAScanConfig


@pytest.fixture
def widget(qt_app):
    from PySide6.QtCore import QSettings
    QSettings("AndorSpectrometer", "TAScanConfig").clear()
    w = TAScanConfigWidget()
    yield w
    w.deleteLater()


class TestTAScanConfigWidgetCreation:
    def test_creates_successfully(self, qt_app):
        w = TAScanConfigWidget()
        assert w is not None
        w.deleteLater()

    def test_has_tab_widget(self, widget):
        from PySide6.QtWidgets import QTabWidget
        tabs = widget.findChildren(QTabWidget)
        assert len(tabs) >= 1

    def test_has_n_averages_field(self, widget):
        assert widget.n_averages_spin is not None

    def test_has_n_scans_field(self, widget):
        assert widget.n_scans_spin is not None

    def test_has_sample_name_field(self, widget):
        assert widget.sample_name_edit is not None

    def test_has_preview_label(self, widget):
        assert widget.preview_label is not None

    def test_has_scan_button(self, widget):
        assert widget.scan_button is not None


class TestTAScanConfigWidgetDefaults:
    def test_default_n_averages(self, widget):
        assert widget.n_averages_spin.value() >= 1

    def test_default_n_scans(self, widget):
        assert widget.n_scans_spin.value() >= 1

    def test_preview_shows_delay_count(self, widget):
        text = widget.preview_label.text()
        assert "delay" in text.lower() or "point" in text.lower()


class TestTAScanConfigWidgetSignal:
    def test_scan_requested_signal_exists(self, widget):
        assert hasattr(widget, "scan_requested")

    def test_scan_button_emits_scan_requested(self, widget):
        emitted = []
        widget.scan_requested.connect(lambda cfg: emitted.append(cfg))
        widget.scan_button.click()
        assert len(emitted) == 1

    def test_scan_requested_emits_ta_scan_config(self, widget):
        emitted = []
        widget.scan_requested.connect(lambda cfg: emitted.append(cfg))
        widget.scan_button.click()
        assert isinstance(emitted[0], TAScanConfig)

    def test_config_uses_n_averages(self, widget):
        widget.n_averages_spin.setValue(5)
        emitted = []
        widget.scan_requested.connect(lambda cfg: emitted.append(cfg))
        widget.scan_button.click()
        assert emitted[0].n_averages == 5

    def test_config_uses_sample_name(self, widget):
        widget.sample_name_edit.setText("my_sample")
        emitted = []
        widget.scan_requested.connect(lambda cfg: emitted.append(cfg))
        widget.scan_button.click()
        assert emitted[0].sample_name == "my_sample"


class TestChopper2x2Mode:
    def _select_chopper_mode(self, widget):
        combo = widget._acq_mode_combo
        idx = combo.findText("chopper_2x2")
        assert idx >= 0, "chopper_2x2 not in acquisition mode combo"
        combo.setCurrentIndex(idx)

    def test_chopper_mode_in_combo(self, widget):
        combo = widget._acq_mode_combo
        texts = [combo.itemText(i) for i in range(combo.count())]
        assert "chopper_2x2" in texts

    def test_chopper_mode_sets_exposure_2ms(self, widget):
        self._select_chopper_mode(widget)
        assert widget._camera_settings.exposure_spin.value() == pytest.approx(0.002)

    def test_chopper_mode_sets_trigger_fast_external(self, widget):
        self._select_chopper_mode(widget)
        trig = widget._camera_settings.trigger_mode_combo.currentData()
        assert trig == "fast_external"

    def test_camera_settings_has_exposure_spinbox(self, widget):
        from PySide6.QtWidgets import QDoubleSpinBox
        assert hasattr(widget._camera_settings, "exposure_spin")
        assert isinstance(widget._camera_settings.exposure_spin, QDoubleSpinBox)

    def test_exposure_in_get_settings(self, widget):
        widget._camera_settings.exposure_spin.setValue(0.005)
        settings = widget._camera_settings.get_settings()
        assert "exposure_time" in settings
        assert settings["exposure_time"] == pytest.approx(0.005)


class TestTAScanConfigWidgetLinearTab:
    def _select_linear_tab(self, widget):
        from PySide6.QtWidgets import QTabWidget
        tabs = widget.findChildren(QTabWidget)[0]
        for i in range(tabs.count()):
            if "linear" in tabs.tabText(i).lower():
                tabs.setCurrentIndex(i)
                return True
        return False

    def test_has_linear_tab(self, widget):
        assert self._select_linear_tab(widget), "No 'Linear' tab found"

    def test_linear_tab_default_start_um(self, widget):
        self._select_linear_tab(widget)
        assert widget.lin_start_spin.value() == pytest.approx(-57000.0)

    def test_linear_tab_default_end_um(self, widget):
        self._select_linear_tab(widget)
        assert widget.lin_end_spin.value() == pytest.approx(-55800.0)

    def test_linear_tab_default_step_um(self, widget):
        self._select_linear_tab(widget)
        assert widget.lin_step_spin.value() == pytest.approx(3.0)

    def test_default_axis(self, widget):
        assert widget.stage_axis_spin.value() == 2

    def test_linear_scan_emits_delay_list(self, widget):
        self._select_linear_tab(widget)
        widget.lin_start_spin.setValue(0.0)
        widget.lin_end_spin.setValue(30.0)
        widget.lin_step_spin.setValue(3.0)
        emitted = []
        widget.scan_requested.connect(lambda cfg: emitted.append(cfg))
        widget.scan_button.click()
        assert len(emitted[0].delay_list) == 11  # 0, 3, 6, ..., 30

    def test_linear_scan_delay_list_monotone(self, widget):
        self._select_linear_tab(widget)
        widget.lin_start_spin.setValue(0.0)
        widget.lin_end_spin.setValue(15.0)
        widget.lin_step_spin.setValue(3.0)
        emitted = []
        widget.scan_requested.connect(lambda cfg: emitted.append(cfg))
        widget.scan_button.click()
        delays = emitted[0].delay_list
        assert all(delays[i] < delays[i + 1] for i in range(len(delays) - 1))

    def test_n_averages_default(self, widget):
        assert widget.n_averages_spin.value() == 100

    def test_n_averages_range_up_to_10000(self, widget):
        widget.n_averages_spin.setValue(10000)
        assert widget.n_averages_spin.value() == 10000


class TestTAScanConfigWidgetStageTab:
    def _select_stage_tab(self, widget):
        from PySide6.QtWidgets import QTabWidget
        tabs = widget.findChildren(QTabWidget)[0]
        for i in range(tabs.count()):
            if "stage" in tabs.tabText(i).lower():
                tabs.setCurrentIndex(i)
                return True
        return False

    def test_has_stage_tab(self, widget):
        assert self._select_stage_tab(widget), "No 'Stage' tab found"

    def test_stage_tab_default_start_um(self, widget):
        self._select_stage_tab(widget)
        assert widget._stg_start.value() == pytest.approx(-57000.0)

    def test_stage_tab_default_step_um(self, widget):
        self._select_stage_tab(widget)
        assert widget._stg_step.value() == pytest.approx(3.0)

    def test_stage_tab_default_n_steps(self, widget):
        self._select_stage_tab(widget)
        assert widget._stg_n_steps.value() == 400

    def test_stage_scan_emits_correct_delay_count(self, widget):
        self._select_stage_tab(widget)
        widget._stg_n_steps.setValue(10)
        emitted = []
        widget.scan_requested.connect(lambda cfg: emitted.append(cfg))
        widget.scan_button.click()
        assert len(emitted[0].delay_list) == 10

    def test_stage_scan_delay_list_monotone(self, widget):
        self._select_stage_tab(widget)
        widget._stg_start.setValue(0.0)
        widget._stg_step.setValue(3.0)
        widget._stg_n_steps.setValue(5)
        emitted = []
        widget.scan_requested.connect(lambda cfg: emitted.append(cfg))
        widget.scan_button.click()
        delays = emitted[0].delay_list
        assert all(delays[i] < delays[i + 1] for i in range(len(delays) - 1))

    def test_stage_info_label_shows_end_position(self, widget):
        self._select_stage_tab(widget)
        widget._stg_start.setValue(0.0)
        widget._stg_step.setValue(10.0)
        widget._stg_n_steps.setValue(11)
        text = widget._stg_info_label.text()
        assert "100.0" in text  # end = 0 + 10*10 = 100 µm

    def test_stage_info_label_shows_ps(self, widget):
        self._select_stage_tab(widget)
        widget._stg_start.setValue(0.0)
        widget._stg_step.setValue(3.0)
        widget._stg_n_steps.setValue(2)
        text = widget._stg_info_label.text()
        assert "ps" in text


class TestScanParameterSummaryFontSizes:
    """Verify that scan parameter summary labels use larger, more readable fonts."""

    def test_preview_label_is_bold_14pt(self, widget):
        ss = widget._preview_label.styleSheet()
        assert "font-size: 14pt" in ss
        assert "font-weight: bold" in ss

    def test_stg_info_label_is_12pt_dark_gray(self, widget):
        ss = widget._stg_info_label.styleSheet()
        assert "font-size: 12pt" in ss
        assert "#555" in ss

    def test_lin_equiv_label_is_12pt_dark_gray(self, widget):
        ss = widget._lin_equiv_label.styleSheet()
        assert "font-size: 12pt" in ss
        assert "#555" in ss

    def test_log_equiv_label_is_12pt_dark_gray(self, widget):
        ss = widget._log_equiv_label.styleSheet()
        assert "font-size: 12pt" in ss
        assert "#555" in ss

    def test_manual_equiv_label_is_12pt_dark_gray(self, widget):
        ss = widget._manual_equiv_label.styleSheet()
        assert "font-size: 12pt" in ss
        assert "#555" in ss


class TestTAScanConfigWidgetYAML:
    def test_save_load_roundtrip(self, widget, tmp_path):
        widget.n_averages_spin.setValue(7)
        widget.sample_name_edit.setText("roundtrip_test")

        path = tmp_path / "config.yaml"
        widget.save_config(str(path))
        assert path.exists()

        widget.n_averages_spin.setValue(1)
        widget.sample_name_edit.setText("")
        widget.load_config(str(path))

        assert widget.n_averages_spin.value() == 7
        assert widget.sample_name_edit.text() == "roundtrip_test"


class TestTAScanConfigWidgetQSettingsBoolPersistence:
    """Regression: QSettings stores bools natively but earlier load code compared
    to the string "true", so save_hdf5/save_spectra/external_trigger checkboxes
    silently reverted to OFF every launch — even though the directory text was
    restored. Result: scans ran without writing files."""

    def _make_widget(self, qt_app):
        w = TAScanConfigWidget()
        return w

    def test_save_hdf5_check_persists_across_widget_recreation(self, qt_app):
        from PySide6.QtCore import QSettings
        QSettings("AndorSpectrometer", "TAScanConfig").clear()
        w1 = self._make_widget(qt_app)
        try:
            w1._save_hdf5_dir_edit.setText("C:/data/h5")
            w1._save_hdf5_check.setChecked(True)
            w1._save_settings()
        finally:
            w1.deleteLater()

        w2 = self._make_widget(qt_app)
        try:
            assert w2._save_hdf5_check.isChecked() is True
            assert w2._save_hdf5_dir_edit.text() == "C:/data/h5"
        finally:
            w2.deleteLater()

    def test_save_spectra_check_persists_across_widget_recreation(self, qt_app):
        from PySide6.QtCore import QSettings
        QSettings("AndorSpectrometer", "TAScanConfig").clear()
        w1 = self._make_widget(qt_app)
        try:
            w1._save_spectra_dir_edit.setText("C:/data/spectra")
            w1._save_spectra_check.setChecked(True)
            w1._save_settings()
        finally:
            w1.deleteLater()

        w2 = self._make_widget(qt_app)
        try:
            assert w2._save_spectra_check.isChecked() is True
            assert w2._save_spectra_dir_edit.text() == "C:/data/spectra"
        finally:
            w2.deleteLater()

    def test_external_trigger_check_persists_across_widget_recreation(self, qt_app):
        from PySide6.QtCore import QSettings
        QSettings("AndorSpectrometer", "TAScanConfig").clear()
        w1 = self._make_widget(qt_app)
        try:
            w1._external_trigger_check.setChecked(True)
            w1._save_settings()
        finally:
            w1.deleteLater()

        w2 = self._make_widget(qt_app)
        try:
            assert w2._external_trigger_check.isChecked() is True
        finally:
            w2.deleteLater()
