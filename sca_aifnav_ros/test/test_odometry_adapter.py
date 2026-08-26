"""Tests for ROS 2 odometry adaptation."""

import math

import pytest
from nav_msgs.msg import Odometry

from sca_aifnav_core.baseline_odometry import (
    CognitiveOdomState,
)
from sca_aifnav_core.planar_geometry import (
    Point2D,
)
from sca_aifnav_ros.odometry_adapter import (
    OdometryAdapter,
)


def odometry_message(
    x,
    y,
):
    """Create a minimal planar ROS 2 odometry message."""
    message = Odometry()

    message.pose.pose.position.x = float(x)
    message.pose.pose.position.y = float(y)

    return message


def test_adapter_starts_uninitialized():
    adapter = OdometryAdapter()

    assert adapter.initialized is False

    assert adapter.state == CognitiveOdomState(
        position=Point2D(
            0.0,
            0.0,
        ),
        travel_heading_rad=0.0,
    )


def test_first_message_establishes_reference_without_motion():
    adapter = OdometryAdapter()

    state = adapter.update(
        odometry_message(
            3.0,
            -2.0,
        )
    )

    assert adapter.initialized is True

    assert state.position == Point2D(
        0.0,
        0.0,
    )

    assert state.travel_heading_rad == pytest.approx(
        0.0
    )


def test_eastward_displacement_has_zero_heading():
    adapter = OdometryAdapter()

    adapter.update(
        odometry_message(
            1.0,
            1.0,
        )
    )

    state = adapter.update(
        odometry_message(
            2.0,
            1.0,
        )
    )

    assert state.position == Point2D(
        1.0,
        0.0,
    )

    assert state.travel_heading_rad == pytest.approx(
        0.0
    )


def test_northward_displacement_has_pi_over_two_heading():
    adapter = OdometryAdapter()

    adapter.update(
        odometry_message(
            1.0,
            1.0,
        )
    )

    state = adapter.update(
        odometry_message(
            1.0,
            2.0,
        )
    )

    assert state.travel_heading_rad == pytest.approx(
        math.pi / 2.0
    )


def test_westward_displacement_has_pi_heading():
    adapter = OdometryAdapter()

    adapter.update(
        odometry_message(
            1.0,
            1.0,
        )
    )

    state = adapter.update(
        odometry_message(
            0.0,
            1.0,
        )
    )

    assert state.travel_heading_rad == pytest.approx(
        math.pi
    )


def test_southward_heading_is_wrapped_to_positive_angle():
    adapter = OdometryAdapter()

    adapter.update(
        odometry_message(
            1.0,
            1.0,
        )
    )

    state = adapter.update(
        odometry_message(
            1.0,
            0.0,
        )
    )

    assert state.travel_heading_rad == pytest.approx(
        3.0 * math.pi / 2.0
    )


def test_robot_body_orientation_is_not_used_as_cognitive_heading():
    adapter = OdometryAdapter()

    first = odometry_message(
        0.0,
        0.0,
    )

    first.pose.pose.orientation.z = 1.0
    first.pose.pose.orientation.w = 0.0

    adapter.update(first)

    second = odometry_message(
        1.0,
        0.0,
    )

    second.pose.pose.orientation.z = 1.0
    second.pose.pose.orientation.w = 0.0

    state = adapter.update(second)

    # The quaternion above represents a body orientation different from
    # the eastward displacement, but cognitive heading follows motion.
    assert state.travel_heading_rad == pytest.approx(
        0.0
    )


def test_reset_requires_a_new_reference_message():
    adapter = OdometryAdapter()

    adapter.update(
        odometry_message(
            2.0,
            3.0,
        )
    )

    adapter.update(
        odometry_message(
            3.0,
            3.0,
        )
    )

    adapter.reset()

    assert adapter.initialized is False

    assert adapter.state.position == Point2D(
        0.0,
        0.0,
    )

    state = adapter.update(
        odometry_message(
            8.0,
            9.0,
        )
    )

    assert state.position == Point2D(
        0.0,
        0.0,
    )

    assert state.travel_heading_rad == pytest.approx(
        0.0
    )


def test_positions_are_relative_to_first_raw_odometry():
    """Later positions should be expressed relative to the first frame."""
    adapter = OdometryAdapter()

    adapter.update(
        odometry_message(
            3.0,
            -2.0,
        )
    )

    state = adapter.update(
        odometry_message(
            4.5,
            -1.5,
        )
    )

    assert state.position == Point2D(
        1.5,
        0.5,
    )


def test_non_odometry_message_is_rejected():
    adapter = OdometryAdapter()

    with pytest.raises(
        TypeError,
        match=(
            "message must be "
            "nav_msgs.msg.Odometry"
        ),
    ):
        adapter.update(
            object()
        )
