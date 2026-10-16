"""HDF5 data writer for transient absorption measurements.

HDF5 structure::

    /wavelengths                  — 1-D float64 array (nm)
    /metadata/
        attrs: sample_name, notes, creation_time,
               n_delays, n_averages, n_scans, acquisition_mode,
               scan_direction, shots_per_frame, software_version,
               exposure_time_s, trigger_mode, hbin, vbin, ...
    /scan_000/
        time_delays               — 1-D float64 (ps), grows per write_point
        delta_signal                  — 2-D float64 (n_delays × n_wavelengths)
    /scan_001/
        ...

Each ``write_point`` call flushes the file immediately for crash protection.

``auto_filename`` generates a timestamped filename.
``export_csv`` converts HDF5 to a flat CSV (one row per delay, averaged over scans).
"""

from __future__ import annotations

import csv
import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Union

import h5py
import numpy as np

if TYPE_CHECKING:
    from andor_qt.ta.scan_config import TAScanConfig


def auto_filename(sample_name: str, base_dir: Union[str, Path]) -> str:
    """Generate a timestamped HDF5 filename.

    Args:
        sample_name: Sample identifier (used in filename).
        base_dir: Directory where the file will be saved.

    Returns:
        Full absolute path string: ``<base_dir>/YYYYMMDD_HHMMSS_TA_<sample_name>.h5``.
    """
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{ts}_TA_{sample_name}.h5"
    return str(Path(base_dir) / filename)


class TADataWriter:
    """Context manager for writing TA scan data to HDF5.

    Usage::

        with TADataWriter(path, wavelengths, sample_name="sample") as writer:
            writer.begin_scan(0)
            writer.write_point(scan_idx=0, delay_ps=1.0, delta_signal=array)

    Or use ``open()`` / ``finalize()`` explicitly.

    Args:
        path: Output HDF5 file path.
        wavelengths: Wavelength array in nm.
        sample_name: Sample name stored in metadata.
        notes: Free-text notes stored in metadata.
        scan_config: Optional ``TAScanConfig`` whose scan parameters are
            written to ``/metadata`` as HDF5 attributes.
        camera_settings: Optional dict of camera settings.  Keys that match
            the known camera attribute names are written to ``/metadata``.
    """

    # Camera setting keys that are written to /metadata when present.
    # The value is the HDF5 attribute name.  ``exposure_time`` is renamed
    # to ``exposure_time_s`` for clarity.
    _CAMERA_SETTING_KEYS: Dict[str, str] = {
        "exposure_time": "exposure_time_s",
        "trigger_mode": "trigger_mode",
        "hbin": "hbin",
        "vbin": "vbin",
        "vs_speed_index": "vs_speed_index",
        "hs_speed_index": "hs_speed_index",
        "amplifier_type": "amplifier_type",
        "em_gain": "em_gain",
        "preamp_gain_index": "preamp_gain_index",
        "read_area_mode": "read_area_mode",
        "baseline_clamp": "baseline_clamp",
        "keep_cleans": "keep_cleans",
    }

    def __init__(
        self,
        path: Union[str, Path],
        wavelengths: np.ndarray,
        sample_name: str = "",
        notes: str = "",
        scan_config: Optional[TAScanConfig] = None,
        camera_settings: Optional[Dict[str, object]] = None,
        hardware_info: Optional[Dict[str, object]] = None,
    ):
        self._path = Path(path)
        self._wavelengths = np.asarray(wavelengths, dtype=np.float64)
        self._sample_name = sample_name
        self._notes = notes
        self._scan_config = scan_config
        self._camera_settings = camera_settings
        self._hardware_info = hardware_info
        self._file: Optional[h5py.File] = None
        self._scan_groups: dict = {}     # scan_idx → h5py.Group
        self._scan_delays: dict = {}     # scan_idx → list of delays
        self._scan_data: dict = {}       # scan_idx → list of delta_signal arrays
        self._scan_stage_pos: dict = {}  # scan_idx → list of stage positions (µm)
        self._scan_pump: dict = {}       # scan_idx → list of pump-ON spectra
        self._scan_ref: dict = {}        # scan_idx → list of pump-OFF spectra

    def open(self) -> None:
        """Open the HDF5 file and write header datasets."""
        self._file = h5py.File(self._path, "w")
        self._file.create_dataset("wavelengths", data=self._wavelengths)
        meta = self._file.create_group("metadata")
        meta.attrs["sample_name"] = self._sample_name
        meta.attrs["notes"] = self._notes
        meta.attrs["creation_time"] = datetime.datetime.now().isoformat()

        # Write scan parameters from TAScanConfig
        if self._scan_config is not None:
            cfg = self._scan_config
            meta.attrs["n_delays"] = len(cfg.delay_list)
            meta.attrs["n_averages"] = cfg.n_averages
            meta.attrs["n_scans"] = cfg.n_scans
            meta.attrs["acquisition_mode"] = cfg.acquisition_mode
            meta.attrs["scan_direction"] = cfg.scan_direction
            meta.attrs["shots_per_frame"] = cfg.shots_per_frame

            # Software version
            try:
                from andor_qt import __version__
                meta.attrs["software_version"] = __version__
            except (ImportError, AttributeError):
                meta.attrs["software_version"] = "0.1.0"

        # Write camera settings (only keys that are present in the dict)
        if self._camera_settings is not None:
            for src_key, attr_name in self._CAMERA_SETTING_KEYS.items():
                if src_key in self._camera_settings:
                    meta.attrs[attr_name] = self._camera_settings[src_key]

        # Write hardware info (spectrograph, camera serial, stage axis, etc.)
        if self._hardware_info is not None:
            hw_grp = self._file.create_group("hardware")
            for key, val in self._hardware_info.items():
                if val is not None:
                    hw_grp.attrs[key] = val

        self._file.flush()

    def begin_scan(self, scan_idx: int) -> None:
        """Create a new scan group in the HDF5 file.

        Args:
            scan_idx: Zero-based scan index.
        """
        if self._file is None:
            self.open()
        group_name = f"scan_{scan_idx:03d}"
        grp = self._file.create_group(group_name)
        self._scan_groups[scan_idx] = grp
        self._scan_delays[scan_idx] = []
        self._scan_data[scan_idx] = []
        self._file.flush()

    def write_point(
        self,
        scan_idx: int,
        delay_ps: float,
        delta_signal: np.ndarray,
        stage_position_um: Optional[float] = None,
        pump_spectrum: Optional[np.ndarray] = None,
        ref_spectrum: Optional[np.ndarray] = None,
    ) -> None:
        """Write one delay point to the current scan.

        Flushes the file after every write for crash protection.

        Args:
            scan_idx: Scan index (must have called ``begin_scan`` first).
            delay_ps: Time delay in picoseconds.
            delta_signal: ΔI/I₀ spectrum at this delay (1-D array, n_wavelengths).
            stage_position_um: Optional stage position in µm.  When provided,
                a ``stage_positions_um`` dataset is maintained alongside
                ``time_delays``.
            pump_spectrum: Optional averaged pump-ON spectrum (raw counts).
            ref_spectrum: Optional averaged pump-OFF reference spectrum (raw counts).
        """
        self._scan_delays[scan_idx].append(float(delay_ps))
        self._scan_data[scan_idx].append(np.asarray(delta_signal, dtype=np.float64))
        if stage_position_um is not None:
            self._scan_stage_pos.setdefault(scan_idx, []).append(float(stage_position_um))
        if pump_spectrum is not None:
            self._scan_pump.setdefault(scan_idx, []).append(
                np.asarray(pump_spectrum, dtype=np.float64)
            )
        if ref_spectrum is not None:
            self._scan_ref.setdefault(scan_idx, []).append(
                np.asarray(ref_spectrum, dtype=np.float64)
            )

        grp = self._scan_groups[scan_idx]
        # Overwrite datasets each time (simplest crash-safe approach)
        delays_arr = np.array(self._scan_delays[scan_idx], dtype=np.float64)
        data_arr = np.array(self._scan_data[scan_idx], dtype=np.float64)

        if "time_delays" in grp:
            del grp["time_delays"]
        if "delta_signal" in grp:
            del grp["delta_signal"]

        grp.create_dataset("time_delays", data=delays_arr)
        grp.create_dataset("delta_signal", data=data_arr)

        if scan_idx in self._scan_stage_pos:
            pos_arr = np.array(self._scan_stage_pos[scan_idx], dtype=np.float64)
            if "stage_positions_um" in grp:
                del grp["stage_positions_um"]
            grp.create_dataset("stage_positions_um", data=pos_arr)

        if scan_idx in self._scan_pump:
            pump_arr = np.array(self._scan_pump[scan_idx], dtype=np.float64)
            if "pump_spectrum" in grp:
                del grp["pump_spectrum"]
            grp.create_dataset("pump_spectrum", data=pump_arr)

        if scan_idx in self._scan_ref:
            ref_arr = np.array(self._scan_ref[scan_idx], dtype=np.float64)
            if "ref_spectrum" in grp:
                del grp["ref_spectrum"]
            grp.create_dataset("ref_spectrum", data=ref_arr)

        self._file.flush()

    def update_metadata(self, attrs: Dict[str, object]) -> None:
        """Write extra attributes to ``/metadata`` (no-op if the file is not open)."""
        if self._file is None:
            return
        meta = self._file["metadata"]
        for key, val in attrs.items():
            meta.attrs[key] = val
        self._file.flush()

    def finalize(self) -> None:
        """Close the HDF5 file."""
        if self._file is not None:
            self._file.close()
            self._file = None

    def __enter__(self) -> "TADataWriter":
        self.open()
        return self

    def __exit__(self, *args) -> None:
        self.finalize()


def export_csv(h5_path: Union[str, Path], output_path: Union[str, Path]) -> None:
    """Export averaged TA data from HDF5 to flat CSV.

    Columns: ``delay_ps``, ``wl_400.0``, ``wl_401.0``, ...

    Rows are averaged ΔOD values across all scans for each unique delay.

    Args:
        h5_path: Source HDF5 file.
        output_path: Destination CSV file.
    """
    with h5py.File(h5_path, "r") as f:
        wavelengths = f["wavelengths"][:]
        scan_keys = sorted(k for k in f.keys() if k.startswith("scan_"))

        if not scan_keys:
            # Write empty CSV with header only
            with open(output_path, "w", newline="", encoding="utf-8") as csvf:
                writer = csv.writer(csvf)
                header = ["delay_ps"] + [f"wl_{wl:.1f}" for wl in wavelengths]
                writer.writerow(header)
            return

        # Collect all data across scans
        delays_all: List[float] = []
        data_all: List[np.ndarray] = []
        for key in scan_keys:
            grp = f[key]
            delays = grp["time_delays"][:]
            data = grp["delta_signal"][:]
            for i, d in enumerate(delays):
                delays_all.append(float(d))
                data_all.append(data[i])

        # Average across scans by delay (simple: assume same delays each scan)
        unique_delays = sorted(set(delays_all))
        delay_to_od: dict = {d: [] for d in unique_delays}
        for delay, od in zip(delays_all, data_all):
            delay_to_od[delay].append(od)

        header = ["delay_ps"] + [f"wl_{wl:.1f}" for wl in wavelengths]
        with open(output_path, "w", newline="", encoding="utf-8") as csvf:
            csv_writer = csv.writer(csvf)
            csv_writer.writerow(header)
            for delay in unique_delays:
                mean_od = np.mean(delay_to_od[delay], axis=0)
                row = [f"{delay:.6g}"] + [f"{v:.8g}" for v in mean_od]
                csv_writer.writerow(row)
