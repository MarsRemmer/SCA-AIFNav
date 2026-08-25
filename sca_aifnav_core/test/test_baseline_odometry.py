"""Tests for the V5-compatible cognitive odometry tracker."""

import math

import pytest

from sca_aifnav_core.baseline_odometry import (
    BaselineOdomTracker,
    CognitiveOdomState,
    wrap_heading_2pi,
)
from sca_aifnav_core.planar_geometry import Point2D


def test_tracker_starts_at_origin():
    tracker = BaselineOdomTracker()

    assert tracker.state.position == Point2D(0.0, 0.0)
    assert tracker.state.travel_heading_rad == pytest.approx(
        0.0
    )


def test_state_can_be_returned_as_tuple():
    state = CognitiveOdomState(
        position=Point2D(1.0, 2.0),
        travel_heading_rad=0.5,
    )

    assert state.as_tuple() == pytest.approx(
        (1.0, 2.0, 0.5)
    )


@pytest.mark.parametrize(
    ("angle", "expected"),
    [
        (0.0, 0.0),
        (math.pi / 2.0, math.pi / 2.0),
        (math.pi, math.pi),
        (3.0 * math.pi / 2.0, 3.0 * math.pi / 2.0),
        (2.0 * math.pi, 0.0),
        (-math.pi / 2.0, 3.0 * math.pi / 2.0),
        (-math.pi, math.pi),
        (5.0 * math.pi / 2.0, math.pi / 2.0),
    ],
)
def test_heading_wrap_matches_v5_360_behavior(
    angle,
    expected,
):
    assert wrap_heading_2pi(angle) == pytest.approx(
        expected
    )


def test_update_east_sets_zero_heading():
    tracker = BaselineOdomTracker()

    state = tracker.update_position(
        Point2D(1.0, 0.0)
    )

    assert state.position == Point2D(1.0, 0.0)
    assert state.travel_heading_rad == pytest.approx(
        0.0
    )


def test_update_north_sets_pi_over_two_heading():
    tracker = BaselineOdomTracker()

    state = tracker.update_position(
        Point2D(0.0, 1.0)
    )

    assert state.travel_heading_rad == pytest.approx(
        math.pi / 2.0
    )


def test_update_west_sets_pi_heading():
    tracker = BaselineOdomTracker()

    state = tracker.update_position(
        Point2D(-1.0, 0.0)
    )

    assert state.travel_heading_rad == pytest.approx(
        math.pi
    )


def test_update_south_sets_three_pi_over_two_heading():
    tracker = BaselineOdomTracker()

    state = tracker.update_position(
        Point2D(0.0, -1.0)
    )

    assert state.travel_heading_rad == pytest.approx(
        3.0 * math.pi / 2.0
    )


def test_heading_uses_previous_position():
    tracker = BaselineOdomTracker()

    tracker.update_position(
        Point2D(2.0, 1.0)
    )

    state = tracker.update_position(
        Point2D(2.0, 3.0)
    )

    assert state.position == Point2D(2.0, 3.0)
    assert state.travel_heading_rad == pytest.approx(
        math.pi / 2.0
    )


def test_same_position_sets_zero_heading_like_atan2_zero_zero():
    tracker = BaselineOdomTracker()

    tracker.reset(
        position=Point2D(2.0, 3.0),
        travel_heading_rad=1.5,
    )

    state = tracker.update_position(
        Point2D(2.0, 3.0)
    )

    assert state.travel_heading_rad == pytest.approx(
        0.0
    )


def test_preview_does_not_modify_tracker():
    tracker = BaselineOdomTracker()

    tracker.reset(
        position=Point2D(1.0, 1.0),
        travel_heading_rad=0.25,
    )

    preview = tracker.preview_position(
        Point2D(1.0, 2.0)
    )

    assert preview.position == Point2D(1.0, 2.0)
    assert preview.travel_heading_rad == pytest.approx(
        math.pi / 2.0
    )

    assert tracker.state.position == Point2D(1.0, 1.0)
    assert tracker.state.travel_heading_rad == pytest.approx(
        0.25
    )


def test_reset_with_position_defaults_heading_to_zero():
    tracker = BaselineOdomTracker()

    state = tracker.reset(
        position=Point2D(4.0, 5.0)
    )

    assert state.position == Point2D(4.0, 5.0)
    assert state.travel_heading_rad == pytest.approx(
        0.0
    )


def test_reset_without_arguments_returns_origin():
    tracker = BaselineOdomTracker()

    tracker.update_position(
        Point2D(3.0, 4.0)
    )

    state = tracker.reset()

    assert state.position == Point2D(0.0, 0.0)
    assert state.travel_heading_rad == pytest.approx(
        0.0
    )


@pytest.mark.parametrize(
    "invalid_angle",
    [
        math.inf,
        -math.inf,
        math.nan,
    ],
)
def test_invalid_heading_is_rejected(invalid_angle):
    with pytest.raises(ValueError):
        CognitiveOdomState(
            position=Point2D(0.0, 0.0),
            travel_heading_rad=invalid_angle,
        )
