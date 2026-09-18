"""Resolve laser-frame mounting orientation from ROS 2 TF."""

import rclpy.time
from tf2_ros import (
    ConnectivityException,
    ExtrapolationException,
    LookupException,
)

from sca_aifnav_ros.orientation_adapter import (
    OrientationAdapter,
)


class LaserFrameTransform:
    """Resolve the laser-frame yaw relative to the robot base frame."""

    def __init__(
        self,
        tf_buffer,
        base_frame_id: str = "base_link",
    ) -> None:
        """Store the TF buffer and robot base-frame identifier."""
        if not isinstance(
            base_frame_id,
            str,
        ):
            raise TypeError(
                "base_frame_id must be a string"
            )

        if not base_frame_id.strip():
            raise ValueError(
                "base_frame_id must not be empty"
            )

        self.tf_buffer = tf_buffer
        self.base_frame_id = (
            base_frame_id.strip()
        )

    def resolve_yaw(
        self,
        laser_frame_id: str,
    ) -> float:
        """Return laser yaw relative to the robot base frame."""
        if not isinstance(
            laser_frame_id,
            str,
        ):
            raise TypeError(
                "laser_frame_id must be a string"
            )

        laser_frame_id = (
            laser_frame_id.strip()
        )

        if not laser_frame_id:
            raise ValueError(
                "laser_frame_id must not be empty"
            )

        try:
            transform = (
                self.tf_buffer.lookup_transform(
                    self.base_frame_id,
                    laser_frame_id,
                    rclpy.time.Time(),
                )
            )
        except (
            LookupException,
            ConnectivityException,
            ExtrapolationException,
        ):
            return None

        return (
            OrientationAdapter
            .yaw_from_quaternion(
                transform
                .transform
                .rotation
            )
        )
