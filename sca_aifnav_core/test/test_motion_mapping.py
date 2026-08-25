"""Tests for geometric mapping of baseline motion primitives."""

import math

import pytest

from sca_aifnav_core.motion_mapping import (
    projected_target,
    target_action,
    target_distance,
)
from sca_aifnav_core.motion_primitives import BaselineMotionSet
from sca_aifnav_core.planar_geometry import PlanarPose, Point2D


@pytest.fixture
def motion_set():
    return BaselineMotionSet()


def test_target_distance():
    pose = PlanarPose.from_xy_yaw(
        0.0,
        0.0,
        0.0,
    )
    target = Point2D(3.0, 4.0)

    assert target_distance(pose, target) == pytest.approx(5.0)


def test_global_east_maps_to_action_zero(motion_set):
    pose = PlanarPose.from_xy_yaw(
        0.0,
        0.0,
        0.0,
    )
    target = Point2D(10.0, 0.0)

    assert target_action(
        pose,
        target,
        motion_set,
    ) == 0


def test_global_north_maps_to_action_three(motion_set):
    pose = PlanarPose.from_xy_yaw(
        0.0,
        0.0,
        0.0,
    )
    target = Point2D(0.0, 10.0)

    assert target_action(
        pose,
        target,
        motion_set,
    ) == 3


def test_global_west_maps_to_action_six(motion_set):
    pose = PlanarPose.from_xy_yaw(
        0.0,
        0.0,
        0.0,
    )
    target = Point2D(-10.0, 0.0)

    assert target_action(
        pose,
        target,
        motion_set,
    ) == 6


def test_global_south_maps_to_action_nine(motion_set):
    pose = PlanarPose.from_xy_yaw(
        0.0,
        0.0,
        0.0,
    )
    target = Point2D(0.0, -10.0)

    assert target_action(
        pose,
        target,
        motion_set,
    ) == 9


def test_action_mapping_is_independent_of_robot_yaw(motion_set):
    pose = PlanarPose.from_xy_yaw(
        0.0,
        0.0,
        math.pi / 2.0,
    )

    target = Point2D(0.0, 10.0)

    assert target_action(
        pose,
        target,
        motion_set,
    ) == 3


def test_projected_action_zero_uses_global_sector_center(motion_set):
    pose = PlanarPose.from_xy_yaw(
        0.0,
        0.0,
        0.0,
    )

    target = projected_target(
        pose=pose,
        action_id=0,
        distance=2.0,
        motion_set=motion_set,
    )

    expected_angle = math.radians(15.0)

    assert target.x == pytest.approx(
        2.0 * math.cos(expected_angle)
    )
    assert target.y == pytest.approx(
        2.0 * math.sin(expected_angle)
    )


def test_projection_is_independent_of_robot_yaw(motion_set):
    pose = PlanarPose.from_xy_yaw(
        0.0,
        0.0,
        math.pi / 2.0,
    )

    target = projected_target(
        pose=pose,
        action_id=0,
        distance=2.0,
        motion_set=motion_set,
    )

    expected_angle = math.radians(15.0)

    assert target.x == pytest.approx(
        2.0 * math.cos(expected_angle)
    )
    assert target.y == pytest.approx(
        2.0 * math.sin(expected_angle)
    )


def test_projection_preserves_pose_offset(motion_set):
    pose = PlanarPose.from_xy_yaw(
        3.0,
        4.0,
        2.0,
    )

    target = projected_target(
        pose=pose,
        action_id=3,
        distance=2.0,
        motion_set=motion_set,
    )

    expected_angle = math.radians(105.0)

    assert target.x == pytest.approx(
        3.0 + 2.0 * math.cos(expected_angle)
    )
    assert target.y == pytest.approx(
        4.0 + 2.0 * math.sin(expected_angle)
    )


def test_stationary_projection_keeps_position(motion_set):
    pose = PlanarPose.from_xy_yaw(
        3.0,
        4.0,
        1.0,
    )

    target = projected_target(
        pose=pose,
        action_id=12,
        distance=100.0,
        motion_set=motion_set,
    )

    assert target == pose.position


@pytest.mark.parametrize(
    "invalid_distance",
    [
        -1.0,
        -0.001,
    ],
)
def test_negative_projection_distance_is_rejected(
    motion_set,
    invalid_distance,
):
    pose = PlanarPose.from_xy_yaw(
        0.0,
        0.0,
        0.0,
    )

    with pytest.raises(ValueError):
        projected_target(
            pose,
            0,
            invalid_distance,
            motion_set,
        )


@pytest.mark.parametrize(
    "invalid_distance",
    [
        math.inf,
        -math.inf,
        math.nan,
    ],
)
def test_non_finite_projection_distance_is_rejected(
    motion_set,
    invalid_distance,
):
    pose = PlanarPose.from_xy_yaw(
        0.0,
        0.0,
        0.0,
    )

    with pytest.raises(ValueError):
        projected_target(
            pose,
            0,
            invalid_distance,
            motion_set,
        )
