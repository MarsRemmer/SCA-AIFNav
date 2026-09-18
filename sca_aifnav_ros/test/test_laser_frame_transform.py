"""Tests for laser-frame TF adaptation."""

import math

import pytest
from geometry_msgs.msg import TransformStamped
from tf2_ros import LookupException

from sca_aifnav_ros.laser_frame_transform import (
    LaserFrameTransform,
)


class FakeTransformBuffer:
    """Provide deterministic transforms without a live ROS graph."""

    def __init__(
        self,
        transform=None,
        error=None,
    ):
        """Store one transform or lookup error."""
        self.transform = transform
        self.error = error
        self.calls = []

    def lookup_transform(
        self,
        target_frame,
        source_frame,
        time,
    ):
        """Record and serve one lookup request."""
        self.calls.append(
            (
                target_frame,
                source_frame,
            )
        )

        if self.error is not None:
            raise self.error

        return self.transform


def transform_with_yaw(
    yaw_rad,
):
    """Create a transform containing one planar yaw."""
    transform = TransformStamped()

    transform.header.frame_id = (
        "base_link"
    )
    transform.child_frame_id = (
        "laser"
    )

    transform.transform.rotation.z = math.sin(
        yaw_rad / 2.0
    )
    transform.transform.rotation.w = math.cos(
        yaw_rad / 2.0
    )

    return transform


def test_laser_yaw_is_read_from_base_to_scan_frame_tf():
    """Read laser mounting yaw from the scan frame TF."""
    buffer = FakeTransformBuffer(
        transform=transform_with_yaw(
            math.pi / 6.0
        )
    )

    resolver = LaserFrameTransform(
        tf_buffer=buffer,
        base_frame_id="base_link",
    )

    yaw = resolver.resolve_yaw(
        laser_frame_id="laser"
    )

    assert yaw == pytest.approx(
        round(
            math.pi / 6.0,
            4,
        )
    )

    assert buffer.calls == [
        (
            "base_link",
            "laser",
        )
    ]


def test_negative_tf_yaw_uses_positive_angular_convention():
    """Match the positive-yaw convention used by AIMAPP."""
    buffer = FakeTransformBuffer(
        transform=transform_with_yaw(
            -math.pi / 6.0
        )
    )

    resolver = LaserFrameTransform(
        tf_buffer=buffer,
        base_frame_id="base_link",
    )

    yaw = resolver.resolve_yaw(
        laser_frame_id="laser"
    )

    assert yaw == pytest.approx(
        round(
            -math.pi / 6.0,
            4,
        )
        + 2.0 * math.pi
    )


def test_scan_frame_is_not_hard_coded_to_laser():
    """Use LaserScan frame identifiers instead of a fixed laser name."""
    buffer = FakeTransformBuffer(
        transform=transform_with_yaw(
            0.0
        )
    )

    resolver = LaserFrameTransform(
        tf_buffer=buffer,
        base_frame_id="base_link",
    )

    resolver.resolve_yaw(
        laser_frame_id="base_scan"
    )

    assert buffer.calls == [
        (
            "base_link",
            "base_scan",
        )
    ]


def test_empty_scan_frame_is_rejected():
    """A LaserScan must identify its source coordinate frame."""
    resolver = LaserFrameTransform(
        tf_buffer=FakeTransformBuffer(),
        base_frame_id="base_link",
    )

    with pytest.raises(
        ValueError,
        match="laser_frame_id",
    ):
        resolver.resolve_yaw(
            laser_frame_id=""
        )


def test_unavailable_tf_returns_none():
    """Unavailable TF should not invent a zero mounting yaw."""
    buffer = FakeTransformBuffer(
        error=LookupException(
            "transform unavailable"
        )
    )

    resolver = LaserFrameTransform(
        tf_buffer=buffer,
        base_frame_id="base_link",
    )

    assert resolver.resolve_yaw(
        laser_frame_id="laser"
    ) is None
