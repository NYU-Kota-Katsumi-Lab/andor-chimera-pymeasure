"""Pytest tests for DelayStage instrument wrapper.

Tests for MockDelayStage and basic interface functionality.
Fixtures (mock_delay_stage, initialized_delay_stage) are provided by conftest.py.
"""

from __future__ import annotations

import threading
import time

import pytest

from andor_pymeasure.instruments.delay_stage import (
    DelayStage,
    DelayStageInfo,
    MockDelayStage,
    SPEED_OF_LIGHT_MM_PS,
)


class TestMockDelayStageInitialization:
    """Tests for MockDelayStage initialization."""

    def test_create_with_defaults(self):
        """MockDelayStage can be created with default parameters."""
        stage = MockDelayStage()
        assert stage._position_min == 0.0
        assert stage._position_max == 300.0
        assert stage._velocity == 10.0

    def test_create_with_custom_params(self):
        """MockDelayStage accepts custom parameters."""
        stage = MockDelayStage(
            position_min=10.0,
            position_max=200.0,
            velocity=50.0,
        )
        assert stage._position_min == 10.0
        assert stage._position_max == 200.0
        assert stage._velocity == 50.0

    def test_initialize_success(self, mock_delay_stage):
        """Delay stage initializes correctly."""
        mock_delay_stage.initialize()

        assert mock_delay_stage._initialized
        assert mock_delay_stage.info is not None

    def test_initialize_sets_info(self, mock_delay_stage):
        """Initialize populates DelayStageInfo."""
        mock_delay_stage.initialize()
        info = mock_delay_stage.info

        assert isinstance(info, DelayStageInfo)
        assert info.model == "Mock Delay Stage"
        assert info.serial_number == "MOCK-001"
        assert info.position_min == 0.0
        assert info.position_max == 100.0

    def test_initialize_already_initialized(self, initialized_delay_stage, caplog):
        """Re-initializing logs a warning."""
        initialized_delay_stage.initialize()
        assert "already initialized" in caplog.text.lower()

    def test_info_none_before_init(self, mock_delay_stage):
        """Info is None before initialization."""
        assert mock_delay_stage.info is None


class TestMockDelayStagePositionMm:
    """Tests for position in mm."""

    def test_position_mm_default(self, initialized_delay_stage):
        """Position starts at 0."""
        assert initialized_delay_stage.position_mm == 0.0

    def test_position_mm_setter(self, initialized_delay_stage):
        """Position can be set."""
        initialized_delay_stage.position_mm = 50.0
        assert initialized_delay_stage.position_mm == 50.0

    def test_position_mm_clamped_to_max(self, initialized_delay_stage):
        """Position is clamped to max limit."""
        initialized_delay_stage.position_mm = 200.0  # Over max of 100
        assert initialized_delay_stage.position_mm == 100.0

    def test_position_mm_clamped_to_min(self, initialized_delay_stage):
        """Position is clamped to min limit."""
        initialized_delay_stage.position_mm = -50.0  # Under min of 0
        assert initialized_delay_stage.position_mm == 0.0

    def test_position_mm_not_initialized_raises(self, mock_delay_stage):
        """Setting position raises if not initialized."""
        with pytest.raises(RuntimeError, match="not initialized"):
            mock_delay_stage.position_mm = 10.0

    def test_position_mm_getter_before_init(self, mock_delay_stage):
        """Getting position before init returns initial value."""
        assert mock_delay_stage.position_mm == 0.0


class TestMockDelayStagePositionPs:
    """Tests for position in ps (optical delay)."""

    def test_position_ps_conversion(self, initialized_delay_stage):
        """Position in ps is correctly converted from mm."""
        initialized_delay_stage.position_mm = 10.0
        # delay_ps = (2 * position_mm) / SPEED_OF_LIGHT_MM_PS
        expected_ps = (2 * 10.0) / SPEED_OF_LIGHT_MM_PS
        assert abs(initialized_delay_stage.position_ps - expected_ps) < 0.001

    def test_position_ps_setter(self, initialized_delay_stage):
        """Position can be set in ps."""
        target_ps = 100.0
        initialized_delay_stage.position_ps = target_ps

        # position_mm = (position_ps * SPEED_OF_LIGHT_MM_PS) / 2
        expected_mm = (target_ps * SPEED_OF_LIGHT_MM_PS) / 2
        assert abs(initialized_delay_stage.position_mm - expected_mm) < 0.001

    def test_position_ps_roundtrip(self, initialized_delay_stage):
        """Setting ps and reading ps returns same value."""
        target_ps = 50.0
        initialized_delay_stage.position_ps = target_ps
        assert abs(initialized_delay_stage.position_ps - target_ps) < 0.001

    def test_delay_range_ps(self, initialized_delay_stage):
        """Delay range in ps is calculated correctly."""
        min_ps, max_ps = initialized_delay_stage.delay_range_ps

        # Expected: (2 * position_min/max) / SPEED_OF_LIGHT_MM_PS
        expected_min = (2 * 0.0) / SPEED_OF_LIGHT_MM_PS
        expected_max = (2 * 100.0) / SPEED_OF_LIGHT_MM_PS

        assert abs(min_ps - expected_min) < 0.001
        assert abs(max_ps - expected_max) < 0.001


class TestMockDelayStageMotion:
    """Tests for motion control."""

    def test_home_moves_to_zero(self, initialized_delay_stage):
        """Home moves stage to position 0."""
        initialized_delay_stage.position_mm = 50.0
        initialized_delay_stage.home()
        assert initialized_delay_stage.position_mm == 0.0

    def test_home_not_initialized_raises(self, mock_delay_stage):
        """Home raises if not initialized."""
        with pytest.raises(RuntimeError, match="not initialized"):
            mock_delay_stage.home()

    def test_stop_halts_motion(self, initialized_delay_stage):
        """Stop clears moving flag."""
        # Set up a long move in background
        initialized_delay_stage._velocity = 0.1  # Slow velocity for long move

        def slow_move():
            try:
                initialized_delay_stage.position_mm = 100.0
            except Exception:
                pass

        thread = threading.Thread(target=slow_move)
        thread.start()

        time.sleep(0.005)  # Let move start (5 ms is enough for thread scheduling)
        initialized_delay_stage.stop()

        # Moving flag should be cleared
        assert not initialized_delay_stage._moving

        # The move itself must end promptly (a lingering non-daemon thread
        # blocks interpreter exit after the test session)
        thread.join(timeout=1.0)
        assert not thread.is_alive(), "stop() did not interrupt the move"
        assert initialized_delay_stage.position_mm < 100.0

    def test_is_moving_during_motion(self, mock_delay_stage):
        """is_moving returns True during motion."""
        mock_delay_stage._velocity = 0.01  # Very slow
        mock_delay_stage.initialize()

        def check_moving():
            # Set moving flag and check
            mock_delay_stage._moving = True
            return mock_delay_stage.is_moving()

        assert not mock_delay_stage.is_moving()

        # Simulate moving
        mock_delay_stage._moving = True
        assert mock_delay_stage.is_moving()


class TestMockDelayStageShutdown:
    """Tests for shutdown."""

    def test_shutdown_success(self, initialized_delay_stage):
        """Shutdown cleans up correctly."""
        initialized_delay_stage.shutdown()
        assert not initialized_delay_stage._initialized

    def test_shutdown_not_initialized_noop(self, mock_delay_stage):
        """Shutdown on non-initialized stage is a no-op."""
        mock_delay_stage.shutdown()
        assert not mock_delay_stage._initialized

    def test_shutdown_stops_motion(self, initialized_delay_stage):
        """Shutdown stops any ongoing motion."""
        initialized_delay_stage._moving = True
        initialized_delay_stage.shutdown()
        assert not initialized_delay_stage._moving


class TestDelayStageConstants:
    """Tests for module constants."""

    def test_speed_of_light_value(self):
        """Speed of light constant is correct."""
        # c = 299,792,458 m/s = 299.792458 mm/ns = 0.299792458 mm/ps
        assert abs(SPEED_OF_LIGHT_MM_PS - 0.299792458) < 1e-9


class TestDelayStageInfo:
    """Tests for DelayStageInfo dataclass."""

    def test_info_frozen(self):
        """DelayStageInfo is immutable."""
        info = DelayStageInfo(
            model="Test",
            serial_number="001",
            position_min=0.0,
            position_max=100.0,
            velocity_max=10.0,
        )

        with pytest.raises(Exception):  # FrozenInstanceError
            info.model = "Changed"

    def test_info_defaults(self):
        """DelayStageInfo has sensible defaults."""
        info = DelayStageInfo()
        assert info.model == ""
        assert info.serial_number == ""
        assert info.position_min == 0.0
        assert info.position_max == 0.0
        assert info.velocity_max == 0.0
