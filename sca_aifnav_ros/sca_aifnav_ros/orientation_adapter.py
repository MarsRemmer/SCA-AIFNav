"""Adapt ROS 2 quaternion orientation to planar robot yaw."""

import math

from geometry_msgs.msg import Quaternion


class OrientationAdapter:
    """Convert ROS 2 quaternion orientation into baseline planar yaw."""

    YAW_DECIMALS = 4

    @classmethod
    def yaw_from_quaternion(
        cls,
        quaternion: Quaternion,
    ) -> float:
        """
        Return physical robot yaw in the baseline angular convention.

        The quaternion is converted directly to yaw, rounded to four
        decimal places, and negative yaw is shifted by 2*pi.
        """
        if not isinstance(
            quaternion,
            Quaternion,
        ):
            raise TypeError(
                "quaternion must be "
                "geometry_msgs.msg.Quaternion"
            )

        components = (
            float(quaternion.x),
            float(quaternion.y),
            float(quaternion.z),
            float(quaternion.w),
        )

        if not all(
            math.isfinite(value)
            for value in components
        ):
            raise ValueError(
                "quaternion components must be finite"
            )

        x, y, z, w = components

        sin_yaw_cos_pitch = (
            2.0
            * (
                w * z
                + x * y
            )
        )

        cos_yaw_cos_pitch = (
            1.0
            - 2.0
            * (
                y * y
                + z * z
            )
        )

        yaw = math.atan2(
            sin_yaw_cos_pitch,
            cos_yaw_cos_pitch,
        )

        yaw = round(
            yaw,
            cls.YAW_DECIMALS,
        )

        if yaw < 0.0:
            yaw += (
                2.0
                * math.pi
            )

        return yaw
