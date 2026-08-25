"""Tests for the baseline discrete motion primitives."""

import math

import pytest

from sca_aifnav_core.motion_primitives import BaselineMotionSet


@pytest.fixture
def motion_set():
    return BaselineMotionSet()


def test_baseline_has_thirteen_actions(motion_set):
    assert len(motion_set) == 13


def test_first_twelve_actions_are_directional(motion_set):
    for action_id in range(12):
        assert motion_set.is_directional(action_id)


def test_action_twelve_is_stationary(motion_set):
    action = motion_set.action(12)

    assert action.action_id == 12
    assert action.is_stationary
    assert not motion_set.is_directional(12)


@pytest.mark.parametrize(
    ("action_id", "expected_center_deg"),
    [
        (0, 15.0),
        (1, 45.0),
        (2, 75.0),
        (3, 105.0),
        (4, 135.0),
        (5, 165.0),
        (6, 195.0),
        (7, 225.0),
        (8, 255.0),
        (9, 285.0),
        (10, 315.0),
        (11, 345.0),
    ],
)
def test_direction_centers_match_v5_baseline(
    motion_set,
    action_id,
    expected_center_deg,
):
    assert motion_set.action(action_id).center_deg == expected_center_deg


def test_center_angle_is_available_in_radians(motion_set):
    assert motion_set.action(0).center_rad == pytest.approx(
        math.radians(15.0)
    )


@pytest.mark.parametrize(
    ("angle_deg", "expected_action"),
    [
        (0.0, 0),
        (10.0, 0),
        (29.999, 0),
        (30.0, 1),
        (45.0, 1),
        (90.0, 3),
        (180.0, 6),
        (359.999, 11),
        (-10.0, 11),
        (-30.0, 11),
        (360.0, 0),
        (390.0, 1),
    ],
)
def test_angle_to_action_mapping(
    motion_set,
    angle_deg,
    expected_action,
):
    assert (
        motion_set.action_for_angle_deg(angle_deg)
        == expected_action
    )


@pytest.mark.parametrize(
    ("action_id", "expected_reverse"),
    [
        (0, 6),
        (1, 7),
        (2, 8),
        (3, 9),
        (4, 10),
        (5, 11),
        (6, 0),
        (7, 1),
        (8, 2),
        (9, 3),
        (10, 4),
        (11, 5),
        (12, 12),
    ],
)
def test_reverse_action_matches_baseline(
    motion_set,
    action_id,
    expected_reverse,
):
    assert (
        motion_set.reverse_action(action_id)
        == expected_reverse
    )


def test_reverse_is_symmetric(motion_set):
    for action_id in range(13):
        reversed_action = motion_set.reverse_action(action_id)

        assert (
            motion_set.reverse_action(reversed_action)
            == action_id
        )


@pytest.mark.parametrize(
    "invalid_action",
    [-1, 13, 100],
)
def test_invalid_action_id_is_rejected(
    motion_set,
    invalid_action,
):
    with pytest.raises(ValueError):
        motion_set.action(invalid_action)


@pytest.mark.parametrize(
    "invalid_action",
    [True, False, 1.0, "1", None],
)
def test_non_integer_action_id_is_rejected(
    motion_set,
    invalid_action,
):
    with pytest.raises(TypeError):
        motion_set.action(invalid_action)


@pytest.mark.parametrize(
    "invalid_angle",
    [
        math.inf,
        -math.inf,
        math.nan,
    ],
)
def test_non_finite_angle_is_rejected(
    motion_set,
    invalid_angle,
):
    with pytest.raises(ValueError):
        motion_set.action_for_angle_deg(invalid_angle)


def test_stationary_action_has_no_direction_error(motion_set):
    with pytest.raises(ValueError):
        motion_set.angular_error_deg(12, 0.0)
