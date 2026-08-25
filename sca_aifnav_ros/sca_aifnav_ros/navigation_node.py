"""ROS 2 navigation node for SCA-AIFNav integration."""

import rclpy
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

    This first integration stage subscribes only to odometry and converts
    ROS 2 messages into the cognitive odometry representation used by the
    pure Python navigation core.
    """

    def __init__(
        self,
        node_name: str = "sca_aifnav_navigation",
    ) -> None:
        """Create the navigation node and odometry subscription."""
        super().__init__(node_name)

        self.declare_parameter(
            "odom_topic",
            "/odom",
        )

        odom_topic = self.get_parameter(
            "odom_topic"
        ).value

        self._odometry_adapter = (
            OdometryAdapter()
        )

        self._latest_odometry_state = None
        self._odometry_revision = 0

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

    def _odometry_callback(
        self,
        message: Odometry,
    ) -> None:
        """Convert and store one ROS 2 odometry message."""
        self._latest_odometry_state = (
            self._odometry_adapter.update(
                message
            )
        )

        self._odometry_revision += 1


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
