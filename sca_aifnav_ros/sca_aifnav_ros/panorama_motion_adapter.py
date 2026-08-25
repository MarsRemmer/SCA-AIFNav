"""ROS motion adaptation for panoramic rotation control."""

from geometry_msgs.msg import Twist

from sca_aifnav_ros.panorama_rotation import (
    PanoramaRotationCommand,
)


class PanoramaMotionAdapter:
    """Convert panorama rotation commands into ROS Twist messages."""

    @staticmethod
    def to_twist(
        command: PanoramaRotationCommand,
    ) -> Twist:
        """Convert one panorama rotation command into a Twist."""
        if not isinstance(
            command,
            PanoramaRotationCommand,
        ):
            raise TypeError(
                "command must be a PanoramaRotationCommand"
            )

        message = Twist()

        if command.goal_reached:
            return message

        message.angular.z = float(
            command.angular_speed_rad_s
        )

        return message

    @staticmethod
    def stop_twist() -> Twist:
        """Return an explicit zero-velocity Twist."""
        return Twist()
