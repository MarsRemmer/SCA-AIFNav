"""ROS 2 navigation node for SCA-AIFNav integration."""

import rclpy
from geometry_msgs.msg import Vector3
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from sca_aifnav_core.baseline_odometry import (
    CognitiveOdomState,
)
from sca_aifnav_ros.odometry_adapter import (
    OdometryAdapter,
)


class NavigationNode(Node):
    """
    Maintain ROS-facing navigation input state.

    The current integration stage subscribes to ROS odometry, converts it
    into cognitive odometry, and publishes a lightweight diagnostic view
    of the resulting cognitive state.
    """

    def __init__(
        self,
        node_name: str = "sca_aifnav_navigation",
    ) -> None:
        """Create the navigation node and ROS interfaces."""
        super().__init__(node_name)

        self.declare_parameter(
            "odom_topic",
            "/odom",
        )

        self.declare_parameter(
            "cognitive_odometry_topic",
            "/sca_aifnav/cognitive_odometry",
        )

        odom_topic = self.get_parameter(
            "odom_topic"
        ).value

        cognitive_odometry_topic = (
            self.get_parameter(
                "cognitive_odometry_topic"
            ).value
        )

        self._odometry_adapter = (
            OdometryAdapter()
        )

        self._latest_odometry_state = None
        self._odometry_revision = 0

        self._cognitive_odometry_publisher = (
            self.create_publisher(
                Vector3,
                cognitive_odometry_topic,
                10,
            )
        )

        self._odom_subscription = (
            self.create_subscription(
                Odometry,
                odom_topic,
                self._odometry_callback,
                qos_profile_sensor_data,
            )
        )

    @property
    def has_odometry(self) -> bool:
        """Return whether at least one odometry message was received."""
        return (
            self._latest_odometry_state
            is not None
        )

    @property
    def latest_odometry_state(
        self,
    ) -> CognitiveOdomState:
        """Return the most recently converted cognitive odometry state."""
        return self._latest_odometry_state

    @property
    def odometry_revision(self) -> int:
        """Return the number of processed odometry messages."""
        return self._odometry_revision

    @staticmethod
    def _cognitive_odometry_message(
        state: CognitiveOdomState,
    ) -> Vector3:
        """
        Convert cognitive odometry into a diagnostic ROS message.

        Vector3 fields are used as:
        x = planar cognitive x
        y = planar cognitive y
        z = travel heading in radians
        """
        message = Vector3()

        message.x = float(
            state.x
        )

        message.y = float(
            state.y
        )

        message.z = float(
            state.travel_heading_rad
        )

        return message

    def _odometry_callback(
        self,
        message: Odometry,
    ) -> None:
        """Convert, store, and publish one odometry update."""
        state = self._odometry_adapter.update(
            message
        )

        self._latest_odometry_state = state

        self._odometry_revision += 1

        diagnostic_message = (
            self._cognitive_odometry_message(
                state
            )
        )

        self._cognitive_odometry_publisher.publish(
            diagnostic_message
        )


def main(args=None) -> None:
    """Run the ROS 2 navigation node."""
    rclpy.init(args=args)

    node = NavigationNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
