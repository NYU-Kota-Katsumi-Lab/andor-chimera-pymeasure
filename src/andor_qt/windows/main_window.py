"""Main window for the Andor Spectrometer Qt GUI.

This is the primary application window that assembles all widgets
and coordinates their interactions.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
from PySide6.QtCore import QObject, Qt, Signal, Slot
from PySide6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QScrollArea,
    QSplitter,
    QStackedWidget,
    QStatusBar,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from andor_pymeasure.instruments.andor_camera import camera_readout_metadata
from andor_qt.core.experiment_queue import ExperimentQueueRunner
from andor_qt.core.hardware_manager import HardwareManager
from andor_qt.core.sequencer_adapter import SequencerAdapter
from andor_qt.core.signals import get_hardware_signals
from andor_qt.widgets.display import ImagePlotWidget, ResultsTableWidget, SpectrumPlotWidget, TraceListWidget
from andor_qt.widgets.hardware import (
    DataSettingsWidget,
    DelayStageControlWidget,
    LogSettingsWidget,
    SpectrographControlWidget,
    TemperatureControlWidget,
)
from andor_qt.widgets.dialogs.shutdown_dialog import ShutdownDialog
from andor_qt.widgets.inputs import DynamicInputsWidget, QueueControlWidget
from andor_qt.widgets.inputs.acquire_control import AcquireControlWidget
from andor_qt.widgets.menu_bar import AndorMenuBar
from andor_qt.windows.ta_panel import TAWindowPanel

log = logging.getLogger(__name__)


class AcquisitionSignals(QObject):
    """Signals for thread-safe acquisition updates."""

    progress = Signal(int, float)  # (exp_id, progress)
    spectrum_ready = Signal(object, object, dict)  # (wavelengths, intensities, params)
    image_ready = Signal(object, object, dict)  # (image, wavelengths, params)
    completed = Signal(int)  # exp_id
    failed = Signal(int, str)  # (exp_id, error_message)
    status = Signal(str)  # status message
    shutdown_progress = Signal(str)  # shutdown progress message
    shutdown_complete = Signal()  # shutdown complete


class AndorSpectrometerWindow(QMainWindow):
    """Main window for Andor spectrometer control.

    Layout:
        ┌─────────────────────────────────────────────────────────────────────┐
        │  Menu Bar                                                           │
        ├─────────────────────────────────────┬───────────────────────────────┤
        │   Control Panel (two columns)       │        Display Panel          │
        │  ┌───────────────┬────────────────┐ │ ┌───────────────────────────┐ │
        │  │ Column 1      │ Column 2       │ │ │ QStackedWidget            │ │
        │  │ (Hardware)    │ (Queue/Data)   │ │ │  - SpectrumPlotWidget(1D) │ │
        │  │───────────────│────────────────│ │ │  - ImagePlotWidget (2D)   │ │
        │  │ TempControl   │ QueueTabs      │ │ └───────────────────────────┘ │
        │  │ SpecControl   │ (Single/Seq)   │ │ ┌───────────────────────────┐ │
        │  │ DynamicInputs │ DataSettings   │ │ │ TraceListWidget           │ │
        │  │ AcquireCtrl   │                │ │ └───────────────────────────┘ │
        │  └───────────────┴────────────────┘ │ ┌───────────────────────────┐ │
        │                                     │ │ ResultsTableWidget        │ │
        │                                     │ └───────────────────────────┘ │
        ├─────────────────────────────────────┴───────────────────────────────┤
        │  Status Bar: Ready | Temp: -60.1C (STABILIZED) | WL: 500nm          │
        └─────────────────────────────────────────────────────────────────────┘
    """

    def __init__(self, parent: QWidget | None = None, config=None):
        super().__init__(parent)

        self.setWindowTitle("Andor Spectrometer Control")
        self.setMinimumSize(1400, 900)
        self.resize(1600, 950)

        self._hw_manager = HardwareManager.instance()
        if config is not None:
            self._hw_manager.set_config(config)
        self._signals = get_hardware_signals()
        self._acq_signals = AcquisitionSignals()

        self._current_exp_id: Optional[int] = None
        self._last_calibration: Optional[np.ndarray] = None
        self._last_data: Optional[np.ndarray] = None

        self._setup_ui()
        self._setup_status_bar()
        self._connect_signals()

        # Start hardware initialization
        self._init_hardware()

    def _setup_ui(self) -> None:
        """Set up the main window UI."""
        # Menu bar
        self._menu_bar = AndorMenuBar(self)
        self.setMenuBar(self._menu_bar)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Top-level tab widget: Spectrometer | TA
        self._main_tabs = QTabWidget()
        outer_layout = QHBoxLayout(central_widget)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.addWidget(self._main_tabs)

        # Spectrometer tab wrapper
        spectrometer_tab = QWidget()
        main_layout = QHBoxLayout(spectrometer_tab)

        # Create main splitter
        self._splitter = QSplitter(Qt.Horizontal)

        # Left panel - Two columns
        left_panel = QWidget()
        left_main_layout = QHBoxLayout(left_panel)
        left_main_layout.setContentsMargins(4, 4, 4, 4)
        left_main_layout.setSpacing(8)

        # Column 1: Hardware controls
        col1_widget = QWidget()
        col1 = QVBoxLayout(col1_widget)
        col1.setContentsMargins(0, 0, 0, 0)

        self._temp_control = TemperatureControlWidget(self._hw_manager)
        col1.addWidget(self._temp_control)

        self._spec_control = SpectrographControlWidget(self._hw_manager)
        col1.addWidget(self._spec_control)

        self._delay_control = DelayStageControlWidget(self._hw_manager)
        col1.addWidget(self._delay_control)

        self._inputs_widget = DynamicInputsWidget()
        col1.addWidget(self._inputs_widget)

        self._acquire_control = AcquireControlWidget()
        col1.addWidget(self._acquire_control)

        col1.addStretch()

        col1_scroll = QScrollArea()
        col1_scroll.setWidget(col1_widget)
        col1_scroll.setWidgetResizable(True)
        col1_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        col1_scroll.setFrameShape(QScrollArea.Shape.NoFrame)

        # Column 2: Queue and data settings
        col2_widget = QWidget()
        col2 = QVBoxLayout(col2_widget)
        col2.setContentsMargins(0, 0, 0, 0)

        # Queue control tabs (Single / Sequence)
        self._queue_control = QueueControlWidget()

        self._queue_runner = ExperimentQueueRunner(self._hw_manager)
        self._sequencer_adapter = SequencerAdapter(
            self._inputs_widget, self._hw_manager, self._queue_runner
        )

        from andor_qt.widgets.inputs.sequencer_widget import FeedbackSequencerWidget

        # SequencerWidget expects attribute names (it maps to display names internally)
        sequencer_inputs = [
            "exposure_time", "center_wavelength", "grating",
            "hbin", "num_accumulations", "delay_position",
        ]
        self._sequencer_widget = FeedbackSequencerWidget(
            inputs=sequencer_inputs,
            parent=self._sequencer_adapter,
        )

        self._queue_tabs = QTabWidget()
        self._queue_tabs.addTab(self._queue_control, "Single")
        self._queue_tabs.addTab(self._sequencer_widget, "Sequence")
        col2.addWidget(self._queue_tabs)

        self._data_settings = DataSettingsWidget()
        col2.addWidget(self._data_settings)

        self._log_settings = LogSettingsWidget()
        col2.addWidget(self._log_settings)

        col2.addStretch()

        left_main_layout.addWidget(col1_scroll, stretch=1)
        left_main_layout.addWidget(col2_widget, stretch=1)

        # Right panel - Display
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(4, 4, 4, 4)

        # Stacked widget for 1D/2D plots
        self._plot_stack = QStackedWidget()

        self._spectrum_plot = SpectrumPlotWidget()
        self._plot_stack.addWidget(self._spectrum_plot)

        self._image_plot = ImagePlotWidget()
        self._plot_stack.addWidget(self._image_plot)

        right_layout.addWidget(self._plot_stack, stretch=2)

        # Trace list for overlay management
        self._trace_list = TraceListWidget()
        right_layout.addWidget(self._trace_list, stretch=0)

        # Results table
        self._results_table = ResultsTableWidget()
        right_layout.addWidget(self._results_table, stretch=1)

        # Add panels to splitter
        self._splitter.addWidget(left_panel)
        self._splitter.addWidget(right_panel)
        self._splitter.setStretchFactor(0, 2)  # Left panel: two columns
        self._splitter.setStretchFactor(1, 3)  # Right panel: larger
        self._splitter.setSizes([550, 650])

        main_layout.addWidget(self._splitter)

        # Add tabs to main tab widget
        self._main_tabs.addTab(spectrometer_tab, "Spectrometer")

        # TA tab
        self._ta_panel = TAWindowPanel(hw_manager=self._hw_manager)
        self._main_tabs.addTab(self._ta_panel, "TA")

    def _setup_status_bar(self) -> None:
        """Set up the status bar."""
        self._status_bar = QStatusBar()
        self.setStatusBar(self._status_bar)

        # Status label
        self._status_label = QLabel("Initializing...")
        self._status_bar.addWidget(self._status_label)

        # Temperature label
        self._temp_status_label = QLabel("Temp: -- °C")
        self._status_bar.addPermanentWidget(self._temp_status_label)

        # Wavelength label
        self._wl_status_label = QLabel("WL: -- nm")
        self._status_bar.addPermanentWidget(self._wl_status_label)

        # Delay position label
        self._delay_status_label = QLabel("Delay: -- ps")
        self._status_bar.addPermanentWidget(self._delay_status_label)

    def _connect_signals(self) -> None:
        """Connect signals between components."""
        # Menu bar signals
        self._menu_bar.exit_requested.connect(self.close)
        self._menu_bar.acquire_requested.connect(self._on_queue_clicked)
        self._menu_bar.abort_requested.connect(self._on_abort_clicked)
        self._menu_bar.create_shortcut_requested.connect(self._on_create_shortcut)
        self._menu_bar.realtime_requested.connect(self._on_realtime_mode)

        # Acquire control signals
        self._acquire_control.acquire_requested.connect(self._on_queue_clicked)
        self._acquire_control.abort_requested.connect(self._on_abort_clicked)

        # Hardware signals
        self._signals.camera_initialized.connect(self._on_camera_initialized)
        self._signals.spectrograph_initialized.connect(self._on_spectrograph_initialized)
        self._signals.temperature_changed.connect(self._on_temperature_changed)
        self._signals.wavelength_changed.connect(self._on_wavelength_changed)
        self._signals.axis_position_changed.connect(self._on_delay_position_changed)
        self._signals.error_occurred.connect(self._on_error)

        # Input signals
        self._inputs_widget.read_mode_changed.connect(self._on_read_mode_changed)

        # Queue control signals
        self._queue_control.queue_clicked.connect(self._on_queue_clicked)
        self._queue_control.abort_clicked.connect(self._on_abort_clicked)

        # Queue runner signals (sequence execution)
        self._queue_runner.spectrum_ready.connect(self._on_spectrum_ready)
        self._queue_runner.image_ready.connect(self._on_image_ready)
        self._queue_runner.procedure_started.connect(self._on_sequence_procedure_started)
        self._queue_runner.procedure_completed.connect(self._on_sequence_procedure_completed)
        self._queue_runner.procedure_failed.connect(self._on_sequence_procedure_failed)
        self._queue_runner.queue_progress.connect(self._on_sequence_progress)
        self._queue_runner.queue_completed.connect(
            lambda: self._queue_control.set_running(False)
        )
        self._queue_runner.queue_completed.connect(
            lambda: self._status_label.setText("Sequence complete")
        )

        # TA module → lock out spectrometer acquire/queue during camera use
        self._ta_panel.camera_busy.connect(self._on_ta_camera_busy)

        # Trace overlay signals
        self._spectrum_plot.trace_added.connect(self._trace_list.add_trace)
        self._trace_list.visibility_toggled.connect(self._spectrum_plot.set_trace_visible)
        self._trace_list.trace_remove_requested.connect(self._spectrum_plot.remove_trace)
        self._trace_list.clear_all_requested.connect(self._spectrum_plot.clear_traces)
        self._spectrum_plot.trace_removed.connect(self._trace_list.remove_trace)
        self._spectrum_plot.traces_cleared.connect(self._trace_list.clear)

        # Acquisition signals (thread-safe)
        self._acq_signals.progress.connect(self._on_acq_progress)
        self._acq_signals.spectrum_ready.connect(self._on_spectrum_ready)
        self._acq_signals.image_ready.connect(self._on_image_ready)
        self._acq_signals.completed.connect(self._on_acq_completed)
        self._acq_signals.failed.connect(self._on_acq_failed)
        self._acq_signals.status.connect(self._on_acq_status)
        self._acq_signals.shutdown_progress.connect(self._on_shutdown_progress)
        self._acq_signals.shutdown_complete.connect(self._on_shutdown_complete)

    def _init_hardware(self) -> None:
        """Initialize hardware in background."""
        self._set_controls_enabled(False)

        self._hw_manager.initialize(
            on_complete=self._on_hardware_ready,
            on_error=self._on_hardware_error,
        )

    @Slot()
    def _on_hardware_ready(self) -> None:
        """Handle hardware initialization complete."""
        log.info("Hardware ready")
        # Emit signal to update UI from main thread
        self._signals.status_message.emit("Ready")

    @Slot(dict)
    def _on_camera_initialized(self, info: dict) -> None:
        """Handle camera initialization."""
        log.info(f"Camera initialized: {info}")

        # Start temperature polling from main thread (QTimer requirement)
        self._hw_manager.start_temperature_polling(interval_ms=2000)

        # Enable controls
        self._status_label.setText("Ready")
        self._set_controls_enabled(True)

        # Inject hardware into procedures
        from andor_qt.procedures import ImageProcedure, SpectrumProcedure

        self._hw_manager.inject_into_procedure(SpectrumProcedure)
        self._hw_manager.inject_into_procedure(ImageProcedure)

        # Populate camera-specific options (pre-amp gains, EM gain range)
        if self._hw_manager.camera:
            self._inputs_widget.populate_from_camera(self._hw_manager.camera)
            self._ta_panel.config_widget.populate_from_camera(self._hw_manager.camera)
            self._ta_panel.monitor_widget.populate_from_camera(self._hw_manager.camera)

    @Slot(str)
    def _on_hardware_error(self, error_msg: str) -> None:
        """Handle hardware initialization error."""
        log.error(f"Hardware error: {error_msg}")
        self._status_label.setText(f"Error: {error_msg}")

        QMessageBox.critical(
            self,
            "Hardware Error",
            f"Failed to initialize hardware:\n\n{error_msg}",
        )

    @Slot(dict)
    def _on_spectrograph_initialized(self, info: dict) -> None:
        """Handle spectrograph initialization."""
        log.info(f"Spectrograph initialized: {info}")

    @Slot(float, str)
    def _on_temperature_changed(self, temperature: float, status: str) -> None:
        """Update temperature status display."""
        self._temp_status_label.setText(f"Temp: {temperature:.1f}°C ({status})")

    @Slot(float)
    def _on_wavelength_changed(self, wavelength: float) -> None:
        """Update wavelength status display."""
        self._wl_status_label.setText(f"WL: {wavelength:.1f} nm")

    @Slot(str, float)
    def _on_delay_position_changed(self, axis_name: str, position: float) -> None:
        """Update delay position status display."""
        # Convert mm to ps for display
        if self._hw_manager.motion_manager:
            axis = self._hw_manager.motion_manager.get_axis(axis_name)
            if axis:
                self._delay_status_label.setText(f"Delay: {axis.position_ps:.2f} ps")

    @Slot(str, str)
    def _on_error(self, source: str, message: str) -> None:
        """Handle error from hardware."""
        log.error(f"Error from {source}: {message}")
        self._status_label.setText(f"Error: {message}")

    @Slot(str)
    def _on_read_mode_changed(self, mode: str) -> None:
        """Handle read mode change."""
        if mode == "image":
            self._plot_stack.setCurrentWidget(self._image_plot)
        else:
            self._plot_stack.setCurrentWidget(self._spectrum_plot)

    # Sequence execution handlers (from queue runner)
    @Slot(int, object)
    def _on_sequence_procedure_started(self, queue_idx: int, procedure) -> None:
        """Handle sequencer procedure started - add to results table."""
        params = {
            "exposure_time": getattr(procedure, "exposure_time", 0),
            "center_wavelength": getattr(procedure, "center_wavelength", 0),
            "grating": getattr(procedure, "grating", 1),
            "delay_position": getattr(procedure, "delay_position", 0),
            "read_mode": "fvb",  # Default; could check procedure type
        }

        # Add to results table
        procedure_type = type(procedure).__name__
        exp_id = self._results_table.add_experiment(procedure_type, params)

        # Store mapping from queue index to experiment ID
        if not hasattr(self, "_queue_exp_map"):
            self._queue_exp_map = {}
        self._queue_exp_map[queue_idx] = exp_id

        self._current_exp_id = exp_id
        self._results_table.update_status(exp_id, "running")
        self._queue_control.set_running(True)
        self._status_label.setText(f"Running procedure {queue_idx + 1}...")

    @Slot(int)
    def _on_sequence_procedure_completed(self, queue_idx: int) -> None:
        """Handle sequencer procedure completed."""
        if hasattr(self, "_queue_exp_map") and queue_idx in self._queue_exp_map:
            exp_id = self._queue_exp_map[queue_idx]
            self._results_table.update_status(exp_id, "completed", 100)

    @Slot(int, str)
    def _on_sequence_procedure_failed(self, queue_idx: int, error_msg: str) -> None:
        """Handle sequencer procedure failed."""
        if hasattr(self, "_queue_exp_map") and queue_idx in self._queue_exp_map:
            exp_id = self._queue_exp_map[queue_idx]
            self._results_table.update_status(exp_id, "failed", error_message=error_msg)
        self._status_label.setText(f"Error: {error_msg}")

    @Slot(int, int)
    def _on_sequence_progress(self, completed: int, total: int) -> None:
        """Handle overall sequence progress."""
        progress = 100 * completed / total if total > 0 else 0
        self._queue_control.set_progress(progress)
        self._status_label.setText(f"Sequence: {completed}/{total} complete")

    # Acquisition signal handlers (called on main thread)
    @Slot(int, float)
    def _on_acq_progress(self, exp_id: int, progress: float) -> None:
        """Handle acquisition progress update."""
        self._queue_control.set_progress(progress)
        self._results_table.update_status(exp_id, "running", progress)

    @Slot(object, object, dict)
    def _on_spectrum_ready(self, wavelengths, intensities, params: dict = None) -> None:
        """Handle spectrum data ready — add as new overlay trace.

        Args:
            wavelengths: Wavelength calibration array.
            intensities: Spectrum intensity data.
            params: Procedure parameters used for this acquisition (from sequencer).
                    If None, falls back to current form values.
        """
        self._last_calibration = wavelengths
        self._last_data = intensities

        # Use passed params if available, otherwise fall back to form values
        if params is None:
            params = self._inputs_widget.get_parameters()

        exp_id = self._current_exp_id or 0
        exp_t = params.get("exposure_time", 0)
        wl = params.get("center_wavelength", 0)
        delay = params.get("delay_position", 0)

        # Build descriptive label with key parameters
        label = f"#{exp_id} exp={exp_t:.2f}s λ={wl:.0f}nm"
        if delay != 0:
            label += f" t={delay:.1f}ps"

        self._spectrum_plot.add_trace(wavelengths, intensities, label=label)

        # Auto-save if enabled
        if self._data_settings.auto_save:
            self._save_data(intensities, wavelengths, params)

    @Slot(object, object, dict)
    def _on_image_ready(self, image, wavelengths, params: dict = None) -> None:
        """Handle image data ready.

        Args:
            image: 2D image data.
            wavelengths: Wavelength calibration array.
            params: Procedure parameters used for this acquisition.
        """
        self._last_calibration = wavelengths
        self._last_data = image

        if params is None:
            params = self._inputs_widget.get_parameters()

        self._image_plot.set_data(image, wavelengths)

        # Auto-save if enabled
        if self._data_settings.auto_save:
            self._save_data(image, wavelengths, params)

    @Slot(int)
    def _on_acq_completed(self, exp_id: int) -> None:
        """Handle acquisition completed."""
        self._results_table.update_status(exp_id, "completed", 100)
        self._queue_control.set_running(False)
        self._status_label.setText("Ready")
        # Note: Auto-save is handled in _on_spectrum_ready / _on_image_ready

    @Slot(int, str)
    def _on_acq_failed(self, exp_id: int, error_msg: str) -> None:
        """Handle acquisition failed."""
        self._results_table.update_status(exp_id, "failed", error_message=error_msg)
        self._queue_control.set_running(False)
        self._status_label.setText(f"Error: {error_msg}")

    @Slot(str)
    def _on_acq_status(self, message: str) -> None:
        """Handle status message from acquisition."""
        self._status_label.setText(message)

    @Slot()
    def _on_queue_clicked(self) -> None:
        """Handle Queue button click - run acquisition."""
        if not self._hw_manager.is_initialized:
            QMessageBox.warning(self, "Not Ready", "Hardware is not initialized.")
            return

        # Get current settings
        params = self._inputs_widget.get_parameters()
        params["center_wavelength"] = self._hw_manager.spectrograph.wavelength
        params["grating"] = self._hw_manager.spectrograph.grating

        # Add to results table
        procedure_type = "FVB" if params["read_mode"] == "fvb" else "Image"
        exp_id = self._results_table.add_experiment(procedure_type, params)
        self._current_exp_id = exp_id

        # Run acquisition in background
        self._queue_control.set_running(True)
        self._results_table.update_status(exp_id, "running")
        self._status_label.setText("Acquiring...")

        import threading

        def _run_acquisition():
            try:
                # Emit status
                self._acq_signals.status.emit("Getting calibration...")

                # Get hbin for calibration
                hbin = params.get("hbin", 1)

                # Get calibration based on source setting
                cal_source = self._data_settings.calibration_source
                if cal_source == "file" and self._data_settings.calibration_file:
                    calibration = self._load_calibration_file(
                        self._data_settings.calibration_file
                    )
                else:
                    calibration = self._hw_manager.get_calibration(hbin=hbin)

                # Apply UI camera settings (amplifier, speeds, gains) so the
                # hardware matches what gets recorded in the metadata
                cam_settings = params.get("camera_settings")
                if cam_settings:
                    self._hw_manager.camera.apply_camera_settings(cam_settings)

                # Set exposure
                self._hw_manager.camera.set_exposure(params["exposure_time"])

                self._acq_signals.status.emit("Acquiring...")

                # Acquire based on mode
                if params["read_mode"] == "fvb":
                    # FVB acquisition with horizontal binning
                    if params.get("num_accumulations", 1) > 1:
                        eff_xpixels = self._hw_manager.camera.xpixels // hbin
                        accumulated = np.zeros(eff_xpixels, dtype=np.float64)
                        for i in range(params["num_accumulations"]):
                            data = self._hw_manager.camera.acquire_fvb(hbin=hbin)
                            accumulated += data
                            progress = 100 * (i + 1) / params["num_accumulations"]
                            self._acq_signals.progress.emit(exp_id, progress)
                        data = accumulated / params["num_accumulations"]
                    else:
                        data = self._hw_manager.camera.acquire_fvb(hbin=hbin)

                    params.update(camera_readout_metadata(self._hw_manager.camera))
                    # Emit spectrum ready (on main thread)
                    self._acq_signals.spectrum_ready.emit(calibration, data, params)
                else:
                    # Image acquisition
                    data = self._hw_manager.camera.acquire_image(
                        hbin=params["hbin"],
                        vbin=params["vbin"],
                    )

                    params.update(camera_readout_metadata(self._hw_manager.camera))
                    # Emit image ready (on main thread)
                    self._acq_signals.image_ready.emit(data, calibration, params)

                # Emit completed
                self._acq_signals.completed.emit(exp_id)

            except Exception as e:
                log.error(f"Acquisition failed: {e}")
                self._acq_signals.failed.emit(exp_id, str(e))

        thread = threading.Thread(target=_run_acquisition, daemon=True)
        thread.start()

    @Slot()
    def _on_abort_clicked(self) -> None:
        """Handle Abort button click."""
        if self._hw_manager.camera:
            self._hw_manager.camera.abort_acquisition()

        # Also abort the queue runner if running
        if self._queue_runner.is_running:
            self._queue_runner.abort_all()

        self._queue_control.set_running(False)
        self._status_label.setText("Aborted")

        if self._current_exp_id:
            self._results_table.update_status(self._current_exp_id, "aborted")

    @Slot()
    def _on_create_shortcut(self) -> None:
        """Handle Create Desktop Shortcut menu action."""
        try:
            from andor_qt.utils.shortcut import create_desktop_shortcut

            shortcut_path = create_desktop_shortcut(name="Andor Spectrometer")
            QMessageBox.information(
                self,
                "Shortcut Created",
                f"Desktop shortcut created:\n{shortcut_path}",
            )
        except Exception as e:
            log.error(f"Failed to create shortcut: {e}")
            QMessageBox.critical(
                self,
                "Error",
                f"Failed to create desktop shortcut:\n{e}",
            )

    @Slot(bool)
    def _on_ta_camera_busy(self, busy: bool) -> None:
        """Lock out spectrometer acquire/queue controls while TA uses the camera."""
        self._acquire_control.set_acquiring(busy)
        self._queue_control.set_running(busy)
        if busy and hasattr(self, "_realtime_window") and self._realtime_window is not None:
            self._realtime_window.stop_acquisition()

    @Slot()
    def _on_realtime_mode(self) -> None:
        """Handle Real-Time Mode menu action."""
        from andor_qt.windows.realtime_window import RealtimeWindow

        if not hasattr(self, "_realtime_window") or self._realtime_window is None:
            self._realtime_window = RealtimeWindow(self._hw_manager, self)
        if self._hw_manager.camera:
            self._realtime_window.populate_from_camera(self._hw_manager.camera)
        self._realtime_window.show()
        self._realtime_window.raise_()
        self._realtime_window.activateWindow()

    def _load_calibration_file(self, filepath: str) -> Optional[np.ndarray]:
        """Load wavelength calibration from a file.

        File format: (pixel, wavelength) pairs, one per line.
        Supports CSV (.csv), text (.txt, .cal) with comma/space separators,
        and NPZ files with a 'wavelengths' array.

        For text files, fits a polynomial (degree 2) to the calibration points
        and generates wavelengths for all pixels.

        Args:
            filepath: Path to the calibration file.

        Returns:
            Numpy array of wavelengths, or None if loading fails.
        """
        try:
            path = Path(filepath)

            if path.suffix.lower() == ".npz":
                # NPZ format - look for 'wavelengths' key (full array)
                data = np.load(filepath)
                if "wavelengths" in data:
                    return data["wavelengths"]
                elif "calibration" in data:
                    return data["calibration"]
                else:
                    log.error(f"NPZ file missing 'wavelengths' key: {filepath}")
                    return None

            elif path.suffix.lower() in (".csv", ".txt", ".cal"):
                # Text format - (pixel, wavelength) pairs
                # Try comma delimiter first, then whitespace
                try:
                    data = np.loadtxt(filepath, delimiter=",", comments="#")
                except ValueError:
                    data = np.loadtxt(filepath, comments="#")

                if data.ndim != 2 or data.shape[1] < 2:
                    log.error(
                        f"Calibration file must have pixel,wavelength columns: {filepath}"
                    )
                    return None

                pixels = data[:, 0]
                wavelengths = data[:, 1]

                # Fit polynomial (degree 2) to calibration points
                coeffs = np.polyfit(pixels, wavelengths, deg=2)

                # Generate wavelengths for all pixels
                num_pixels = 1024  # Default
                if self._hw_manager.camera:
                    num_pixels = self._hw_manager.camera.xpixels
                all_pixels = np.arange(num_pixels)
                calibration = np.polyval(coeffs, all_pixels)

                log.info(
                    f"Loaded calibration from {filepath}: "
                    f"{len(data)} points, fitted polynomial"
                )
                return calibration

            log.error(f"Unsupported calibration file format: {filepath}")
            return None

        except Exception as e:
            log.error(f"Failed to load calibration file: {e}")
            return None

    def _save_data(
        self,
        data: np.ndarray,
        calibration: Optional[np.ndarray],
        params: dict,
    ) -> None:
        """Save acquisition data to file.

        Supports two metadata modes:
        - 'separate': Creates clean data file + .meta.json sidecar
        - 'embedded': Stores metadata in file headers (legacy)

        Args:
            data: Acquired data (1D or 2D).
            calibration: Wavelength calibration array.
            params: Acquisition parameters.
        """
        try:
            use_separate = self._data_settings.metadata_format == "separate"
            session_meta = self._data_settings.get_metadata()

            if data.ndim == 1:
                # 1D spectrum - save as CSV
                filepath = self._data_settings.get_next_filepath(".csv")

                if use_separate:
                    from andor_qt.utils.data_io import save_csv_data_only
                    from andor_qt.utils.metadata import save_metadata

                    if calibration is not None:
                        save_csv_data_only(filepath, calibration, data)
                    else:
                        # Fallback: use pixel indices as x-axis
                        wavelengths = np.arange(len(data), dtype=float)
                        save_csv_data_only(filepath, wavelengths, data)

                    save_metadata(filepath, params, session_meta)
                else:
                    from andor_qt.utils.data_io import save_csv_with_metadata

                    if calibration is not None:
                        save_csv_with_metadata(
                            filepath, calibration, data, params, session_meta
                        )
                    else:
                        wavelengths = np.arange(len(data), dtype=float)
                        save_csv_with_metadata(
                            filepath, wavelengths, data, params, session_meta
                        )

            else:
                # 2D image - save as NPZ
                filepath = self._data_settings.get_next_filepath(".npz")

                if use_separate:
                    from andor_qt.utils.data_io import save_npz_data_only
                    from andor_qt.utils.metadata import save_metadata

                    save_npz_data_only(filepath, calibration, data)
                    save_metadata(filepath, params, session_meta)
                else:
                    from andor_qt.utils.data_io import save_npz_with_metadata

                    save_npz_with_metadata(
                        filepath, calibration, data, params, session_meta
                    )

            log.info(f"Data saved to {filepath}")
            self._status_label.setText(f"Saved: {filepath.name}")

        except Exception as e:
            log.error(f"Failed to save data: {e}")

    def _set_controls_enabled(self, enabled: bool) -> None:
        """Enable or disable control widgets."""
        self._temp_control.set_enabled(enabled)
        self._spec_control.set_enabled(enabled)
        self._delay_control.set_enabled(enabled)
        self._inputs_widget.setEnabled(enabled)
        self._queue_control.setEnabled(enabled)

    def closeEvent(self, event) -> None:
        """Handle window close - shutdown hardware safely."""
        # Force quit bypasses warmup (QApplication.quit re-triggers closeEvent)
        if getattr(self, "_force_quitting", False):
            event.accept()
            return

        # Check if hardware is initialized and needs warmup
        if not self._hw_manager.is_initialized:
            event.accept()
            return

        # Check current temperature
        current_temp = None
        if self._hw_manager.camera:
            try:
                current_temp = self._hw_manager.camera.temperature
            except Exception:
                pass

        # Build confirmation message
        if current_temp is not None and current_temp < -20:
            msg = (
                f"Are you sure you want to exit?\n\n"
                f"Camera temperature: {current_temp:.1f}°C\n"
                f"The camera will warm up to -20°C before shutdown.\n"
                f"This may take a few minutes."
            )
        else:
            msg = "Are you sure you want to exit?"

        reply = QMessageBox.question(
            self,
            "Confirm Exit",
            msg,
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )

        if reply != QMessageBox.Yes:
            event.ignore()
            return

        # Prevent the close and start shutdown with visual progress
        event.ignore()
        self._start_shutdown_with_dialog()

    def _start_shutdown(self) -> None:
        """Start the shutdown process.

        Note: This blocks the UI during shutdown/warmup. This is intentional
        to avoid Qt crashes from cross-thread signal/timer interactions.
        """
        import threading

        self._set_controls_enabled(False)
        self._status_label.setText("Shutting down (please wait)...")

        # Force UI update before blocking
        QApplication.processEvents()

        # Use an Event to wait for the hardware manager's internal
        # shutdown thread to actually complete (not just the call returning).
        shutdown_done = threading.Event()

        self._hw_manager.shutdown(
            warmup=True,
            on_complete=lambda: shutdown_done.set(),
            on_progress=lambda msg: log.info(f"Shutdown: {msg}"),
        )

        # Block until the hardware shutdown thread finishes
        shutdown_done.wait(timeout=360)

        log.info("Shutdown complete, closing application")
        QApplication.instance().quit()

    def _start_shutdown_with_dialog(self) -> None:
        """Start shutdown with a progress dialog.

        Shows a ShutdownDialog that displays temperature progress
        and allows force-quit.
        """
        self._set_controls_enabled(False)

        # Get current temperature for progress range
        start_temp = -60.0
        if self._hw_manager.camera:
            try:
                start_temp = self._hw_manager.camera.temperature
            except Exception:
                pass

        target_temp = -20.0

        # Create and show dialog
        self._shutdown_dialog = ShutdownDialog(self)
        self._shutdown_dialog.set_temperature_range(start_temp, target_temp)
        self._shutdown_dialog.set_status("Starting shutdown...")
        self._shutdown_dialog.force_quit_requested.connect(self._on_force_quit)

        # Connect temperature signal to update dialog
        self._signals.temperature_changed.connect(
            self._shutdown_dialog.update_temperature
        )

        self._shutdown_dialog.show()

        # Start shutdown in background
        self._hw_manager.shutdown(
            warmup=True,
            on_complete=lambda: self._acq_signals.shutdown_complete.emit(),
            on_progress=lambda msg: self._acq_signals.shutdown_progress.emit(msg),
        )

    def _on_force_quit(self) -> None:
        """Handle force quit request from shutdown dialog."""
        log.warning("Force quit requested - bypassing warmup")
        self._force_quitting = True
        QApplication.instance().quit()

    @Slot()
    def _update_shutdown_progress(self) -> None:
        """Update shutdown progress bar based on last received temperature.

        Note: Does NOT access hardware directly to avoid race conditions.
        Temperature updates come from _on_shutdown_progress signal.
        """
        if not hasattr(self, "_shutdown_dialog") or self._shutdown_dialog is None:
            return

        # Calculate progress from stored temperature
        temp = getattr(self, "_shutdown_last_temp", self._shutdown_start_temp)
        temp_range = self._shutdown_target_temp - self._shutdown_start_temp
        if abs(temp_range) > 0.1:
            progress = (temp - self._shutdown_start_temp) / temp_range * 100
            progress = max(0, min(100, progress))
        else:
            progress = 100

        self._shutdown_dialog.setValue(int(progress))

    @Slot(str)
    def _on_shutdown_progress(self, message: str) -> None:
        """Handle shutdown progress message from hardware manager."""
        log.info(f"Shutdown progress: {message}")
        self._status_label.setText(message)

        # Update shutdown dialog status if visible
        if hasattr(self, "_shutdown_dialog") and self._shutdown_dialog is not None:
            self._shutdown_dialog.set_status(message)

    @Slot()
    def _on_shutdown_complete(self) -> None:
        """Handle shutdown complete - close the application."""
        log.info("Shutdown complete, closing application")
        self._status_label.setText("Shutdown complete")
        self._shutdown_in_progress = False

        # Close the shutdown dialog if visible
        if hasattr(self, "_shutdown_dialog") and self._shutdown_dialog is not None:
            self._shutdown_dialog.on_shutdown_complete()

        # Now close the application
        QApplication.instance().quit()
