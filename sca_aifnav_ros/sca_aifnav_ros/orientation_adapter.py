"""Adapt ROS 2 quaternion orientation to planar robot yaw."""

import math

from geometry_msgs.msg import Quaternion


class OrientationAdapter:
    """Convert ROS 2 quaternion orientation into planar yaw conventions."""

    YAW_DECIMALS = 4

    @classmethod
    def signed_yaw_from_quaternion(
        cls,
        quaternion: Quaternion,
    ) -> float:
        """Return rounded signed physical robot yaw."""
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

        return round(
            yaw,
            cls.YAW_DECIMALS,
        )

    @staticmethod
    def positive_yaw(
        signed_yaw_rad: float,
    ) -> float:
        """Convert signed yaw into the positive angular convention."""
        if not math.isfinite(
            signed_yaw_rad
        ):
            raise ValueError(
                "signed_yaw_rad must be finite"
            )

        yaw = float(
            signed_yaw_rad
        )

        if yaw < 0.0:
            yaw += (
                2.0
                * math.pi
            )

        return yaw

    @classmethod
    def yaw_from_quaternion(
        cls,
        quaternion: Quaternion,
    ) -> float:
        """Return yaw in the positive angular convention."""
        signed_yaw = (
            cls.signed_yaw_from_quaternion(
                quaternion
            )
        )

        return cls.positive_yaw(
            signed_yaw
        )
