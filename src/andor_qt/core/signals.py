"""Qt signals for cross-thread communication.

This module defines signals used for communicating hardware state changes
and events across threads in the Qt GUI.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtCore import QObject, Signal

if TYPE_CHECKING:
    pass


class HardwareSignals(QObject):
    """Qt signals for hardware state changes.

    These signals allow thread-safe communication between background
    hardware operations and the GUI.
    """

    # Camera signals
    camera_initialized = Signal(dict)  # {xpixels, ypixels, serial, etc.}
    camera_shutdown = Signal()
    temperature_changed = Signal(float, str)  # (temperature, status)
    cooler_state_changed = Signal(bool, int)  # (on, target_temp)

    # Acquisition signals
    acquisition_started = Signal()
    acquisition_progress = Signal(int)  # percentage
    acquisition_completed = Signal(object)  # numpy array data
    acquisition_error = Signal(str)  # error message
    acquisition_aborted = Signal()

    # Spectrograph signals
    spectrograph_initialized = Signal(dict)  # {serial, num_gratings, etc.}
    spectrograph_shutdown = Signal()
    grating_changing = Signal(int)  # target grating
    grating_changed = Signal(int)  # new grating
    wavelength_changing = Signal(float)  # target wavelength
    wavelength_changed = Signal(float)  # new wavelength
    calibration_updated = Signal(object)  # wavelength array

    # Motion controller signals
    motion_initialized = Signal(dict)  # {axis_name: info, ...}
    axis_position_changing = Signal(str, float)  # (axis_name, target_position)
    axis_position_changed = Signal(str, float)  # (axis_name, current_position)
    axis_moving = Signal(str, bool)  # (axis_name, is_moving)

    # Chopper signals
    choppers_initialized = Signal(dict)  # {pump_port, probe_port, pump_connected, probe_connected}

    # General signals
    error_occurred = Signal(str, str)  # (source, message)
    status_message = Signal(str)  # status bar message


class ProcedureSignals(QObject):
    """Qt signals for procedure execution.

    These signals communicate procedure state for the experiment queue.
    """

    # Procedure lifecycle
    procedure_queued = Signal(int)  # procedure_id
    procedure_started = Signal(int)  # procedure_id
    procedure_progress = Signal(int, float)  # (procedure_id, progress %)
    procedure_finished = Signal(int)  # procedure_id
    procedure_failed = Signal(int, str)  # (procedure_id, error_message)
    procedure_aborted = Signal(int)  # procedure_id

    # Results
    results_available = Signal(int, object)  # (procedure_id, Results object)


# Global singleton instances
_hardware_signals = None
_procedure_signals = None


def get_hardware_signals() -> HardwareSignals:
    """Get the global HardwareSignals instance."""
    global _hardware_signals
    if _hardware_signals is None:
        _hardware_signals = HardwareSignals()
    return _hardware_signals


def get_procedure_signals() -> ProcedureSignals:
    """Get the global ProcedureSignals instance."""
    global _procedure_signals
    if _procedure_signals is None:
        _procedure_signals = ProcedureSignals()
    return _procedure_signals
