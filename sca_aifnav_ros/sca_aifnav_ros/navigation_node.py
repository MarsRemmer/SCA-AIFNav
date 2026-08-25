"""ROS 2 navigation node for SCA-AIFNav integration."""

import rclpy
from geometry_msgs.msg import Vector3
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32MultiArray

from sca_aifnav_core.baseline_odometry import (
    CognitiveOdomState,
)
from sca_aifnav_ros.obstacle_scan_adapter import (
    ObstacleScanAdapter,
)
from sca_aifnav_ros.odometry_adapter import (
    OdometryAdapter,
)
from sca_aifnav_ros.orientation_adapter import (
    OrientationAdapter,
)


class NavigationNode(Node):
    """
    Maintain ROS-facing navigation input state.

    Odometry provides both cognitive travel state and physical robot yaw.
    Laser scans are transformed into twelve world-frame directional
    obstacle distances once physical orientation is available.
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
            "scan_topic",
            "/scan",
        )

        self.declare_parameter(
            "cognitive_odometry_topic",
            "/sca_aifnav/cognitive_odometry",
        )

        self.declare_parameter(
            "obstacle_distances_topic",
            "/sca_aifnav/obstacle_distances",
        )

        self.declare_parameter(
            "laser_yaw_offset_rad",
            0.0,
        )

        odom_topic = self.get_parameter(
            "odom_topic"
        ).value

        scan_topic = self.get_parameter(
            "scan_topic"
        ).value

        cognitive_odometry_topic = (
            self.get_parameter(
                "cognitive_odometry_topic"
            ).value
        )

        obstacle_distances_topic = (
            self.get_parameter(
                "obstacle_distances_topic"
            ).value
        )

        self._laser_yaw_offset_rad = float(
            self.get_parameter(
                "laser_yaw_offset_rad"
            ).value
        )

        self._odometry_adapter = (
            OdometryAdapter()
        )

        self._obstacle_scan_adapter = (
            ObstacleScanAdapter()
        )

        self._latest_odometry_state = None
        self._latest_physical_yaw_rad = None
        self._latest_obstacle_distances = None

        self._odometry_revision = 0
        self._scan_revision = 0

        self._cognitive_odometry_publisher = (
            self.create_publisher(
                Vector3,
                cognitive_odometry_topic,
                10,
            )
        )

        self._obstacle_distance_publisher = (
            self.create_publisher(
                Float32MultiArray,
                obstacle_distances_topic,
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

        self._scan_subscription = (
            self.create_subscription(
                LaserScan,
                scan_topic,
                self._scan_callback,
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
    def physical_yaw_rad(self):
        """Return the latest physical robot body yaw in radians."""
        return self._latest_physical_yaw_rad

    @property
    def has_obstacle_distances(self) -> bool:
        """Return whether at least one laser scan was processed."""
        return (
            self._latest_obstacle_distances
            is not None
        )

    @property
    def latest_obstacle_distances(self):
        """Return the latest twelve directional obstacle distances."""
        return self._latest_obstacle_distances

    @property
    def odometry_revision(self) -> int:
        """Return the number of processed odometry messages."""
        return self._odometry_revision

    @property
    def scan_revision(self) -> int:
        """Return the number of processed laser scans."""
        return self._scan_revision

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

    @staticmethod
    def _obstacle_distances_message(
        distances,
    ) -> Float32MultiArray:
        """Convert directional obstacle distances into a ROS message."""
        message = Float32MultiArray()

        message.data = [
            float(distance)
            for distance in distances
        ]

        return message

    def _odometry_callback(
        self,
        message: Odometry,
    ) -> None:
        """Convert, store, and publish one odometry update."""
        state = self._odometry_adapter.update(
            message
        )

        physical_yaw_rad = (
            OrientationAdapter.yaw_from_quaternion(
                message.pose.pose.orientation
            )
        )

        self._latest_odometry_state = state
        self._latest_physical_yaw_rad = (
            physical_yaw_rad
        )

        self._odometry_revision += 1

        diagnostic_message = (
            self._cognitive_odometry_message(
                state
            )
        )

        self._cognitive_odometry_publisher.publish(
            diagnostic_message
        )

    def _scan_callback(
        self,
        message: LaserScan,
    ) -> None:
        """Convert and publish one laser scan when orientation is known."""
        if self._latest_physical_yaw_rad is None:
            return

        distances = (
            self._obstacle_scan_adapter.aggregate(
                message,
                robot_yaw_rad=(
                    self._latest_physical_yaw_rad
                ),
                laser_yaw_offset_rad=(
                    self._laser_yaw_offset_rad
                ),
            )
        )

        self._latest_obstacle_distances = distances
        self._scan_revision += 1

        diagnostic_message = (
            self._obstacle_distances_message(
                distances
            )
        )

        self._obstacle_distance_publisher.publish(
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
