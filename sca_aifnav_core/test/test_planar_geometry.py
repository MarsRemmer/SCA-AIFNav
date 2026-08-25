"""Tests for planar geometry utilities."""

import math

import pytest

from sca_aifnav_core.planar_geometry import (
    PlanarPose,
    Point2D,
    wrap_angle_deg,
    wrap_angle_rad,
)


@pytest.mark.parametrize(
    ("angle_deg", "expected_deg"),
    [
        (0.0, 0.0),
        (90.0, 90.0),
        (180.0, -180.0),
        (270.0, -90.0),
        (360.0, 0.0),
        (-90.0, -90.0),
        (-180.0, -180.0),
        (-270.0, 90.0),
        (540.0, -180.0),
    ],
)
def test_wrap_angle_deg(angle_deg, expected_deg):
    assert wrap_angle_deg(angle_deg) == pytest.approx(expected_deg)


@pytest.mark.parametrize(
    ("angle_rad", "expected_rad"),
    [
        (0.0, 0.0),
        (math.pi / 2.0, math.pi / 2.0),
        (math.pi, -math.pi),
        (2.0 * math.pi, 0.0),
        (-math.pi, -math.pi),
        (3.0 * math.pi, -math.pi),
    ],
)
def test_wrap_angle_rad(angle_rad, expected_rad):
    assert wrap_angle_rad(angle_rad) == pytest.approx(expected_rad)


def test_point_distance_uses_euclidean_geometry():
    point_a = Point2D(0.0, 0.0)
    point_b = Point2D(3.0, 4.0)

    assert point_a.distance_to(point_b) == pytest.approx(5.0)


@pytest.mark.parametrize(
    ("target", "expected_bearing"),
    [
        (Point2D(1.0, 0.0), 0.0),
        (Point2D(0.0, 1.0), math.pi / 2.0),
        (Point2D(-1.0, 0.0), math.pi),
        (Point2D(0.0, -1.0), -math.pi / 2.0),
    ],
)
def test_point_bearing_matches_cartesian_axes(
    target,
    expected_bearing,
):
    origin = Point2D(0.0, 0.0)

    assert origin.bearing_to(target) == pytest.approx(
        expected_bearing
    )


def test_point_translation():
    point = Point2D(1.0, 2.0)

    moved = point.translated(3.0, -1.0)

    assert moved == Point2D(4.0, 1.0)


def test_planar_pose_normalizes_yaw():
    pose = PlanarPose.from_xy_yaw(
        x=1.0,
        y=2.0,
        yaw_rad=3.0 * math.pi,
    )

    assert pose.yaw_rad == pytest.approx(-math.pi)


def test_pose_distance_ignores_heading():
    pose_a = PlanarPose.from_xy_yaw(
        0.0,
        0.0,
        0.0,
    )
    pose_b = PlanarPose.from_xy_yaw(
        3.0,
        4.0,
        math.pi,
    )

    assert pose_a.distance_to(pose_b) == pytest.approx(5.0)


def test_relative_bearing_accounts_for_robot_heading():
    pose = PlanarPose.from_xy_yaw(
        0.0,
        0.0,
        math.pi / 2.0,
    )
    target = Point2D(0.0, 10.0)

    assert pose.global_bearing_to(target) == pytest.approx(
        math.pi / 2.0
    )
    assert pose.relative_bearing_to(target) == pytest.approx(0.0)


def test_relative_bearing_right_of_robot():
    pose = PlanarPose.from_xy_yaw(
        0.0,
        0.0,
        math.pi / 2.0,
    )
    target = Point2D(10.0, 0.0)

    assert pose.relative_bearing_to(target) == pytest.approx(
        -math.pi / 2.0
    )


def test_local_forward_motion_at_zero_yaw():
    pose = PlanarPose.from_xy_yaw(
        1.0,
        2.0,
        0.0,
    )

    moved = pose.moved_local(forward=2.0)

    assert moved.position == Point2D(3.0, 2.0)
    assert moved.yaw_rad == pytest.approx(0.0)


def test_local_forward_motion_at_ninety_degrees():
    pose = PlanarPose.from_xy_yaw(
        1.0,
        2.0,
        math.pi / 2.0,
    )

    moved = pose.moved_local(forward=2.0)

    assert moved.x == pytest.approx(1.0)
    assert moved.y == pytest.approx(4.0)


def test_local_lateral_motion():
    pose = PlanarPose.from_xy_yaw(
        0.0,
        0.0,
        math.pi / 2.0,
    )

    moved = pose.moved_local(
        forward=0.0,
        lateral=1.0,
    )

    assert moved.x == pytest.approx(-1.0)
    assert moved.y == pytest.approx(0.0)


@pytest.mark.parametrize(
    "invalid_value",
    [
        math.inf,
        -math.inf,
        math.nan,
    ],
)
def test_non_finite_point_coordinate_is_rejected(
    invalid_value,
):
    with pytest.raises(ValueError):
        Point2D(invalid_value, 0.0)


@pytest.mark.parametrize(
    "invalid_value",
    [
        math.inf,
        -math.inf,
        math.nan,
    ],
)
def test_non_finite_yaw_is_rejected(invalid_value):
    with pytest.raises(ValueError):
        PlanarPose.from_xy_yaw(
            0.0,
            0.0,
            invalid_value,
        )
