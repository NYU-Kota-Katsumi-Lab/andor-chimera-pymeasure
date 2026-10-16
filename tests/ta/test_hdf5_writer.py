"""Tests for TADataWriter (HDF5 output) and CSV export."""

from __future__ import annotations

import csv
import re
from pathlib import Path

import numpy as np
import pytest

try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False

from andor_qt.ta.hdf5_writer import TADataWriter, auto_filename, export_csv
from andor_qt.ta.scan_config import TAScanConfig


pytestmark = pytest.mark.skipif(not HAS_H5PY, reason="h5py not installed")


# ---------------------------------------------------------------------------
# auto_filename
# ---------------------------------------------------------------------------


class TestAutoFilename:
    def test_contains_sample_name(self, tmp_path):
        name = auto_filename("rhodamine_6g", tmp_path)
        assert "rhodamine_6g" in name

    def test_contains_TA(self, tmp_path):
        name = auto_filename("sample", tmp_path)
        assert "TA" in name

    def test_has_h5_extension(self, tmp_path):
        name = auto_filename("sample", tmp_path)
        assert name.endswith(".h5")

    def test_timestamp_format(self, tmp_path):
        name = Path(auto_filename("s", tmp_path)).name
        # YYYYMMDD_HHMMSS_TA_s.h5
        assert re.match(r"\d{8}_\d{6}_TA_", name)


# ---------------------------------------------------------------------------
# TADataWriter
# ---------------------------------------------------------------------------


class TestTADataWriter:
    def test_context_manager_creates_file(self, tmp_path):
        path = tmp_path / "test.h5"
        wavelengths = np.linspace(400.0, 800.0, 64)
        with TADataWriter(path, wavelengths=wavelengths, sample_name="test") as writer:
            pass
        assert path.exists()

    def test_writes_wavelengths_dataset(self, tmp_path):
        path = tmp_path / "test.h5"
        wavelengths = np.linspace(400.0, 800.0, 64)
        with TADataWriter(path, wavelengths=wavelengths, sample_name="test") as writer:
            pass
        with h5py.File(path, "r") as f:
            assert "wavelengths" in f
            assert np.allclose(f["wavelengths"][:], wavelengths)

    def test_writes_sample_name_metadata(self, tmp_path):
        path = tmp_path / "test.h5"
        wavelengths = np.linspace(400.0, 800.0, 10)
        with TADataWriter(path, wavelengths=wavelengths, sample_name="mytest") as writer:
            pass
        with h5py.File(path, "r") as f:
            assert f["metadata"].attrs["sample_name"] == "mytest"

    def test_begin_scan_creates_group(self, tmp_path):
        path = tmp_path / "test.h5"
        wavelengths = np.linspace(400.0, 800.0, 10)
        with TADataWriter(path, wavelengths=wavelengths, sample_name="s") as writer:
            writer.begin_scan(0)
        with h5py.File(path, "r") as f:
            assert "scan_000" in f

    def test_write_point_stores_data(self, tmp_path):
        path = tmp_path / "test.h5"
        wavelengths = np.linspace(400.0, 800.0, 10)
        delta_signal = np.random.rand(10)
        with TADataWriter(path, wavelengths=wavelengths, sample_name="s") as writer:
            writer.begin_scan(0)
            writer.write_point(scan_idx=0, delay_ps=1.0, delta_signal=delta_signal)
        with h5py.File(path, "r") as f:
            assert "scan_000" in f
            assert "time_delays" in f["scan_000"]
            assert "delta_signal" in f["scan_000"]
            assert f["scan_000/time_delays"][0] == pytest.approx(1.0)
            assert np.allclose(f["scan_000/delta_signal"][0], delta_signal)

    def test_multiple_points_per_scan(self, tmp_path):
        path = tmp_path / "test.h5"
        wavelengths = np.linspace(400.0, 800.0, 10)
        delays = [0.5, 1.0, 5.0, 10.0]
        with TADataWriter(path, wavelengths=wavelengths, sample_name="s") as writer:
            writer.begin_scan(0)
            for d in delays:
                writer.write_point(0, d, np.ones(10) * d)
        with h5py.File(path, "r") as f:
            assert len(f["scan_000/time_delays"]) == len(delays)

    def test_multiple_scans(self, tmp_path):
        path = tmp_path / "test.h5"
        wavelengths = np.linspace(400.0, 800.0, 5)
        with TADataWriter(path, wavelengths=wavelengths, sample_name="s") as writer:
            for scan in range(3):
                writer.begin_scan(scan)
                writer.write_point(scan, 1.0, np.ones(5))
        with h5py.File(path, "r") as f:
            assert "scan_000" in f
            assert "scan_001" in f
            assert "scan_002" in f

    def test_write_point_stores_stage_position_um(self, tmp_path):
        path = tmp_path / "test.h5"
        wavelengths = np.linspace(400.0, 800.0, 10)
        with TADataWriter(path, wavelengths=wavelengths, sample_name="s") as writer:
            writer.begin_scan(0)
            writer.write_point(0, 1.0, np.ones(10), stage_position_um=-57000.0)
        with h5py.File(path, "r") as f:
            assert "stage_positions_um" in f["scan_000"]
            assert f["scan_000/stage_positions_um"][0] == pytest.approx(-57000.0)

    def test_write_point_no_stage_position_no_dataset(self, tmp_path):
        path = tmp_path / "test.h5"
        wavelengths = np.linspace(400.0, 800.0, 5)
        with TADataWriter(path, wavelengths=wavelengths, sample_name="s") as writer:
            writer.begin_scan(0)
            writer.write_point(0, 1.0, np.ones(5))
        with h5py.File(path, "r") as f:
            assert "stage_positions_um" not in f["scan_000"]

    def test_write_point_stores_pump_and_ref_spectra(self, tmp_path):
        """Pump-ON and pump-OFF spectra are saved alongside delta_signal."""
        path = tmp_path / "test.h5"
        wavelengths = np.linspace(400.0, 800.0, 10)
        pump = np.ones(10) * 1200.0
        ref = np.ones(10) * 1000.0
        delta = (pump - ref) / ref
        with TADataWriter(path, wavelengths=wavelengths, sample_name="s") as writer:
            writer.begin_scan(0)
            writer.write_point(0, 1.0, delta, pump_spectrum=pump, ref_spectrum=ref)
        with h5py.File(path, "r") as f:
            assert "pump_spectrum" in f["scan_000"]
            assert "ref_spectrum" in f["scan_000"]
            np.testing.assert_allclose(f["scan_000/pump_spectrum"][0], pump)
            np.testing.assert_allclose(f["scan_000/ref_spectrum"][0], ref)

    def test_write_point_no_pump_ref_no_datasets(self, tmp_path):
        """When pump/ref not provided, datasets are absent."""
        path = tmp_path / "test.h5"
        wavelengths = np.linspace(400.0, 800.0, 5)
        with TADataWriter(path, wavelengths=wavelengths, sample_name="s") as writer:
            writer.begin_scan(0)
            writer.write_point(0, 1.0, np.ones(5))
        with h5py.File(path, "r") as f:
            assert "pump_spectrum" not in f["scan_000"]
            assert "ref_spectrum" not in f["scan_000"]

    def test_write_point_pump_ref_multiple_points(self, tmp_path):
        """Pump/ref accumulate across multiple write_point calls."""
        path = tmp_path / "test.h5"
        wavelengths = np.linspace(400.0, 800.0, 5)
        with TADataWriter(path, wavelengths=wavelengths, sample_name="s") as writer:
            writer.begin_scan(0)
            for i in range(3):
                pump = np.ones(5) * (1000.0 + i * 100)
                ref = np.ones(5) * 1000.0
                writer.write_point(0, float(i), np.ones(5) * 0.1,
                                   pump_spectrum=pump, ref_spectrum=ref)
        with h5py.File(path, "r") as f:
            assert f["scan_000/pump_spectrum"].shape == (3, 5)
            assert f["scan_000/ref_spectrum"].shape == (3, 5)
            assert f["scan_000/pump_spectrum"][2, 0] == pytest.approx(1200.0)

    def test_open_close_explicit(self, tmp_path):
        path = tmp_path / "test.h5"
        wavelengths = np.linspace(400.0, 800.0, 5)
        writer = TADataWriter(path, wavelengths=wavelengths, sample_name="s")
        writer.open()
        writer.begin_scan(0)
        writer.write_point(0, 1.0, np.ones(5))
        writer.finalize()
        assert path.exists()


# ---------------------------------------------------------------------------
# export_csv
# ---------------------------------------------------------------------------


class TestExportCSV:
    def _make_h5(self, tmp_path, n_wavelengths=5, n_delays=3, n_scans=2):
        path = tmp_path / "data.h5"
        wavelengths = np.linspace(400.0, 800.0, n_wavelengths)
        with TADataWriter(path, wavelengths=wavelengths, sample_name="test") as writer:
            for scan in range(n_scans):
                writer.begin_scan(scan)
                for i in range(n_delays):
                    delay = float(i + 1)
                    writer.write_point(scan, delay, np.ones(n_wavelengths) * delay)
        return path

    def test_creates_csv(self, tmp_path):
        h5_path = self._make_h5(tmp_path)
        csv_path = tmp_path / "out.csv"
        export_csv(h5_path, csv_path)
        assert csv_path.exists()

    def test_csv_has_header(self, tmp_path):
        h5_path = self._make_h5(tmp_path)
        csv_path = tmp_path / "out.csv"
        export_csv(h5_path, csv_path)
        with open(csv_path) as f:
            reader = csv.reader(f)
            header = next(reader)
        assert "delay_ps" in header[0]

    def test_csv_column_count(self, tmp_path):
        n_wl = 5
        h5_path = self._make_h5(tmp_path, n_wavelengths=n_wl)
        csv_path = tmp_path / "out.csv"
        export_csv(h5_path, csv_path)
        with open(csv_path) as f:
            reader = csv.reader(f)
            header = next(reader)
        # delay_ps + n_wavelengths columns
        assert len(header) == 1 + n_wl

    def test_csv_empty_scans(self, tmp_path):
        """HDF5 with wavelengths but no scans should produce header-only CSV."""
        import h5py
        h5_path = tmp_path / "empty.h5"
        with h5py.File(h5_path, "w") as f:
            f.create_dataset("wavelengths", data=np.linspace(400, 800, 5))
        csv_path = tmp_path / "out.csv"
        export_csv(h5_path, csv_path)
        with open(csv_path) as f:
            lines = f.readlines()
        assert len(lines) == 1  # header only


class TestHDF5WriterEdgeCases:
    def test_begin_scan_auto_opens(self, tmp_path):
        """begin_scan should auto-open the file if not already opened."""
        path = tmp_path / "auto_open.h5"
        wavelengths = np.linspace(400, 800, 5)
        writer = TADataWriter(path, wavelengths=wavelengths, sample_name="test")
        # Do NOT call writer.open() — begin_scan should handle it
        writer.begin_scan(0)
        writer.write_point(0, 1.0, np.ones(5))
        writer.finalize()
        import h5py
        with h5py.File(path, "r") as f:
            assert "scan_000" in f

    def test_write_point_overwrites_stage_positions(self, tmp_path):
        """Multiple write_point calls should overwrite stage_positions_um."""
        path = tmp_path / "overwrite.h5"
        wavelengths = np.linspace(400, 800, 5)
        with TADataWriter(path, wavelengths=wavelengths, sample_name="test") as writer:
            writer.begin_scan(0)
            writer.write_point(0, 1.0, np.ones(5), stage_position_um=100.0)
            writer.write_point(0, 2.0, np.ones(5), stage_position_um=200.0)
        import h5py
        with h5py.File(path, "r") as f:
            positions = f["scan_000/stage_positions_um"][:]
            assert len(positions) == 2
            np.testing.assert_allclose(positions, [100.0, 200.0])


# ---------------------------------------------------------------------------
# Acquisition metadata in /metadata group
# ---------------------------------------------------------------------------


def _make_scan_config(**overrides) -> TAScanConfig:
    """Create a TAScanConfig with sensible defaults for testing."""
    defaults = dict(
        delay_list=[0.0, 1.0, 5.0],
        n_averages=50,
        n_scans=3,
        acquisition_mode="chopper_2x2",
        scan_direction="forward",
        sample_name="rhodamine_6g",
        notes="test measurement",
        shots_per_frame=2,
    )
    defaults.update(overrides)
    return TAScanConfig(**defaults)


_SAMPLE_CAMERA_SETTINGS = {
    "exposure_time": 0.0004,
    "trigger_mode": "fast_external",
    "hbin": 1,
    "vbin": 1,
    "vs_speed_index": 0,
    "hs_speed_index": 0,
    "amplifier_type": 1,
    "em_gain": 1,
    "preamp_gain_index": 2,
    "read_area_mode": "full",
}


class TestAcquisitionMetadata:
    """Test that /metadata group includes scan parameters, camera settings,
    and software version when scan_config and camera_settings are provided."""

    def test_scan_parameters_written(self, tmp_path):
        """Scan parameters from TAScanConfig should appear in /metadata attrs."""
        path = tmp_path / "meta.h5"
        config = _make_scan_config()
        wavelengths = np.linspace(400, 800, 10)
        with TADataWriter(
            path, wavelengths=wavelengths,
            sample_name=config.sample_name, notes=config.notes,
            scan_config=config,
        ) as writer:
            pass
        with h5py.File(path, "r") as f:
            meta = f["metadata"]
            assert meta.attrs["n_delays"] == 3
            assert meta.attrs["n_averages"] == 50
            assert meta.attrs["n_scans"] == 3
            assert meta.attrs["acquisition_mode"] == "chopper_2x2"
            assert meta.attrs["scan_direction"] == "forward"
            assert meta.attrs["shots_per_frame"] == 2

    def test_scan_parameter_types(self, tmp_path):
        """Verify correct types for scan parameter attributes."""
        path = tmp_path / "types.h5"
        config = _make_scan_config()
        wavelengths = np.linspace(400, 800, 10)
        with TADataWriter(
            path, wavelengths=wavelengths,
            sample_name=config.sample_name, notes=config.notes,
            scan_config=config,
        ) as writer:
            pass
        with h5py.File(path, "r") as f:
            meta = f["metadata"]
            assert isinstance(meta.attrs["n_delays"], (int, np.integer))
            assert isinstance(meta.attrs["n_averages"], (int, np.integer))
            assert isinstance(meta.attrs["n_scans"], (int, np.integer))
            assert isinstance(meta.attrs["acquisition_mode"], str)
            assert isinstance(meta.attrs["scan_direction"], str)
            assert isinstance(meta.attrs["shots_per_frame"], (int, np.integer))

    def test_camera_settings_written(self, tmp_path):
        """Camera settings dict should appear in /metadata attrs."""
        path = tmp_path / "cam.h5"
        config = _make_scan_config()
        wavelengths = np.linspace(400, 800, 10)
        with TADataWriter(
            path, wavelengths=wavelengths,
            sample_name=config.sample_name, notes=config.notes,
            scan_config=config,
            camera_settings=_SAMPLE_CAMERA_SETTINGS,
        ) as writer:
            pass
        with h5py.File(path, "r") as f:
            meta = f["metadata"]
            assert meta.attrs["exposure_time_s"] == pytest.approx(0.0004)
            assert meta.attrs["trigger_mode"] == "fast_external"
            assert meta.attrs["hbin"] == 1
            assert meta.attrs["vbin"] == 1
            assert meta.attrs["vs_speed_index"] == 0
            assert meta.attrs["hs_speed_index"] == 0
            assert meta.attrs["amplifier_type"] == 1
            assert meta.attrs["em_gain"] == 1
            assert meta.attrs["preamp_gain_index"] == 2
            assert meta.attrs["read_area_mode"] == "full"

    def test_camera_settings_types(self, tmp_path):
        """Verify correct types for camera settings attributes."""
        path = tmp_path / "cam_types.h5"
        config = _make_scan_config()
        wavelengths = np.linspace(400, 800, 10)
        with TADataWriter(
            path, wavelengths=wavelengths,
            sample_name=config.sample_name, notes=config.notes,
            scan_config=config,
            camera_settings=_SAMPLE_CAMERA_SETTINGS,
        ) as writer:
            pass
        with h5py.File(path, "r") as f:
            meta = f["metadata"]
            assert isinstance(meta.attrs["exposure_time_s"], (float, np.floating))
            assert isinstance(meta.attrs["trigger_mode"], str)
            assert isinstance(meta.attrs["hbin"], (int, np.integer))
            assert isinstance(meta.attrs["vbin"], (int, np.integer))
            assert isinstance(meta.attrs["vs_speed_index"], (int, np.integer))
            assert isinstance(meta.attrs["hs_speed_index"], (int, np.integer))
            assert isinstance(meta.attrs["amplifier_type"], (int, np.integer))
            assert isinstance(meta.attrs["em_gain"], (int, np.integer))
            assert isinstance(meta.attrs["preamp_gain_index"], (int, np.integer))
            assert isinstance(meta.attrs["read_area_mode"], str)

    def test_missing_camera_settings_no_crash(self, tmp_path):
        """When camera_settings is None, no camera attrs should exist but no crash."""
        path = tmp_path / "no_cam.h5"
        config = _make_scan_config()
        wavelengths = np.linspace(400, 800, 10)
        with TADataWriter(
            path, wavelengths=wavelengths,
            sample_name=config.sample_name, notes=config.notes,
            scan_config=config,
        ) as writer:
            pass
        with h5py.File(path, "r") as f:
            meta = f["metadata"]
            # Scan params should still be present
            assert meta.attrs["n_delays"] == 3
            # Camera-specific attrs should NOT be present
            assert "exposure_time_s" not in meta.attrs
            assert "trigger_mode" not in meta.attrs

    def test_no_scan_config_backward_compatible(self, tmp_path):
        """When scan_config is None (old API), only basic metadata is written."""
        path = tmp_path / "compat.h5"
        wavelengths = np.linspace(400, 800, 10)
        with TADataWriter(
            path, wavelengths=wavelengths,
            sample_name="compat_test", notes="backward compat",
        ) as writer:
            pass
        with h5py.File(path, "r") as f:
            meta = f["metadata"]
            assert meta.attrs["sample_name"] == "compat_test"
            assert meta.attrs["notes"] == "backward compat"
            assert "creation_time" in meta.attrs
            # Scan-specific attrs should NOT be present
            assert "n_delays" not in meta.attrs
            assert "n_averages" not in meta.attrs

    def test_software_version_written(self, tmp_path):
        """Software version should appear in /metadata attrs."""
        path = tmp_path / "ver.h5"
        config = _make_scan_config()
        wavelengths = np.linspace(400, 800, 10)
        with TADataWriter(
            path, wavelengths=wavelengths,
            sample_name=config.sample_name, notes=config.notes,
            scan_config=config,
        ) as writer:
            pass
        with h5py.File(path, "r") as f:
            meta = f["metadata"]
            assert "software_version" in meta.attrs
            assert isinstance(meta.attrs["software_version"], str)
            # Should be a valid version string like "0.1.0"
            assert len(meta.attrs["software_version"]) > 0

    def test_partial_camera_settings(self, tmp_path):
        """Only provided camera setting keys should be written."""
        path = tmp_path / "partial.h5"
        config = _make_scan_config()
        wavelengths = np.linspace(400, 800, 10)
        partial_settings = {"exposure_time": 0.001, "trigger_mode": "internal"}
        with TADataWriter(
            path, wavelengths=wavelengths,
            sample_name=config.sample_name, notes=config.notes,
            scan_config=config,
            camera_settings=partial_settings,
        ) as writer:
            pass
        with h5py.File(path, "r") as f:
            meta = f["metadata"]
            assert meta.attrs["exposure_time_s"] == pytest.approx(0.001)
            assert meta.attrs["trigger_mode"] == "internal"
            # Keys not in the dict should be absent
            assert "hbin" not in meta.attrs
            assert "vs_speed_index" not in meta.attrs


class TestUpdateMetadata:
    def test_update_metadata_writes_attrs(self, tmp_path):
        path = tmp_path / "meta.h5"
        writer = TADataWriter(path, wavelengths=np.array([500.0, 600.0]))
        writer.open()
        writer.update_metadata({
            "amplifier_type_applied": 1,
            "readout_flip_corrected": 0,
            "readout_orientation": "short_wavelength_first",
        })
        writer.finalize()

        with h5py.File(path, "r") as f:
            meta = f["metadata"].attrs
            assert meta["amplifier_type_applied"] == 1
            assert meta["readout_flip_corrected"] == 0
            assert meta["readout_orientation"] == "short_wavelength_first"

    def test_update_metadata_before_open_is_noop(self, tmp_path):
        writer = TADataWriter(tmp_path / "x.h5", wavelengths=np.array([500.0]))
        writer.update_metadata({"amplifier_type_applied": 1})  # must not raise
