"""Adapt ROS 2 laser scans to SCA-AIFNav directional obstacle distances."""

import math

from sensor_msgs.msg import LaserScan


class ObstacleScanAdapter:
    """
    Aggregate a 360-degree laser scan into twelve world-frame directions.

    Each laser ray is rotated from the laser frame into the world frame
    using the robot body yaw and an optional laser-to-base yaw offset.

    Rays are grouped into twelve 30-degree action sectors. The grouping
    preserves the baseline closed-interval, first-match behavior at exact
    sector boundaries.

    Positive infinity and NaN range values are replaced by range_max
    before each directional arithmetic mean is computed.
    """

    DIRECTION_COUNT = 12
    SECTOR_WIDTH_DEG = 30.0

    @classmethod
    def action_for_world_angle_deg(
        cls,
        angle_deg: float,
    ) -> int:
        """
        Map one world-frame angle to its directional action.

        Sector boundaries preserve first-match closed-interval behavior:
        30 degrees belongs to action 0,
        60 degrees belongs to action 1, and so on.
        """
        if not math.isfinite(angle_deg):
            raise ValueError(
                "angle_deg must be finite"
            )

        normalized_deg = (
            angle_deg % 360.0
        )

        for action_id in range(
            cls.DIRECTION_COUNT
        ):
            lower = (
                action_id
                * cls.SECTOR_WIDTH_DEG
            )

            upper = (
                lower
                + cls.SECTOR_WIDTH_DEG
            )

            if (
                normalized_deg >= lower
                and normalized_deg <= upper
            ):
                return action_id

        raise RuntimeError(
            "normalized angle did not match an action sector"
        )

    def aggregate(
        self,
        message: LaserScan,
        robot_yaw_rad: float,
        laser_yaw_offset_rad: float = 0.0,
    ) -> list:
        """
        Convert one LaserScan into twelve world-frame obstacle distances.

        Parameters
        ----------
        message:
            ROS 2 laser scan message.
        robot_yaw_rad:
            Physical robot body yaw in the world frame.
        laser_yaw_offset_rad:
            Yaw rotation from the laser frame to the robot base frame.

        """
        if not isinstance(
            message,
            LaserScan,
        ):
            raise TypeError(
                "message must be "
                "sensor_msgs.msg.LaserScan"
            )

        if not math.isfinite(
            robot_yaw_rad
        ):
            raise ValueError(
                "robot_yaw_rad must be finite"
            )

        if not math.isfinite(
            laser_yaw_offset_rad
        ):
            raise ValueError(
                "laser_yaw_offset_rad must be finite"
            )

        distances_by_action = [
            []
            for _ in range(
                self.DIRECTION_COUNT
            )
        ]

        for index, raw_distance in enumerate(
            message.ranges
        ):
            distance = float(
                raw_distance
            )

            if (
                distance == math.inf
                or math.isnan(distance)
            ):
                distance = float(
                    message.range_max
                )

            local_ray_angle_rad = (
                float(message.angle_min)
                + index
                * float(
                    message.angle_increment
                )
            )

            world_ray_angle_rad = (
                robot_yaw_rad
                + laser_yaw_offset_rad
                + local_ray_angle_rad
            )

            world_ray_angle_deg = (
                math.degrees(
                    world_ray_angle_rad
                )
                % 360.0
            )

            action_id = (
                self.action_for_world_angle_deg(
                    world_ray_angle_deg
                )
            )

            distances_by_action[
                action_id
            ].append(
                distance
            )

        obstacle_distances = []

        for distances in distances_by_action:
            if not distances:
                obstacle_distances.append(
                    float("nan")
                )
                continue

            obstacle_distances.append(
                sum(distances)
                / len(distances)
            )

        return obstacle_distances
