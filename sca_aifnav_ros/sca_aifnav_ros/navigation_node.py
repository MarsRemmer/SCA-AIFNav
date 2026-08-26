"""ROS 2 navigation node for SCA-AIFNav integration."""

import rclpy
from geometry_msgs.msg import Twist, Vector3
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, LaserScan
from std_msgs.msg import Float32MultiArray

from sca_aifnav_core.baseline_odometry import (
    CognitiveOdomState,
)
from sca_aifnav_core.motion_primitives import (
    BaselineMotionSet,
)
from sca_aifnav_core.spatial_memory import (
    BaselinePlaceMemory,
)
from sca_aifnav_ros.image_adapter import (
    ImageAdapter,
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
from sca_aifnav_ros.panorama_coordinator import (
    PanoramaCoordinator,
    PanoramaCoordinatorState,
)
from sca_aifnav_ros.panorama_motion_adapter import (
    PanoramaMotionAdapter,
)
from sca_aifnav_ros.panorama_rotation import (
    PanoramaRotationController,
)
from sca_aifnav_ros.sensor_snapshot import (
    capture_sensor_snapshot as build_sensor_snapshot,
)
from sca_aifnav_ros.visual_observation import (
    PanoramicVisualObserver,
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
            "camera_topic",
            "/camera_front/image_raw",
        )

        self.declare_parameter(
            "left_camera_topic",
            "/camera_left/image_raw",
        )

        self.declare_parameter(
            "right_camera_topic",
            "/camera_right/image_raw",
        )

        self.declare_parameter(
            "cmd_vel_topic",
            "/cmd_vel",
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

        self.declare_parameter(
            "panorama_control_period_sec",
            0.05,
        )

        odom_topic = self.get_parameter(
            "odom_topic"
        ).value

        scan_topic = self.get_parameter(
            "scan_topic"
        ).value

        cmd_vel_topic = self.get_parameter(
            "cmd_vel_topic"
        ).value

        camera_topic = self.get_parameter(
            "camera_topic"
        ).value

        left_camera_topic = self.get_parameter(
            "left_camera_topic"
        ).value

        right_camera_topic = self.get_parameter(
            "right_camera_topic"
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

        panorama_control_period_sec = float(
            self.get_parameter(
                "panorama_control_period_sec"
            ).value
        )

        if panorama_control_period_sec <= 0.0:
            raise ValueError(
                "panorama_control_period_sec must be positive"
            )

        self._odometry_adapter = (
            OdometryAdapter()
        )

        self._obstacle_scan_adapter = (
            ObstacleScanAdapter()
        )

        self._image_adapter = (
            ImageAdapter()
        )

        self._panorama_rotation_controller = (
            PanoramaRotationController()
        )

        self._panorama_motion_adapter = (
            PanoramaMotionAdapter()
        )

        self._panorama_coordinator = None

        self._visual_observer = (
            PanoramicVisualObserver()
        )
        self._latest_visual_observation = None

        self._place_memory = (
            BaselinePlaceMemory()
        )
        self._latest_place_observation_id = None

        self._latest_odometry_state = None
        self._latest_physical_yaw_rad = None
        self._latest_obstacle_distances = None
        self._latest_image = None
        self._latest_left_image = None
        self._latest_right_image = None

        self._odometry_revision = 0
        self._scan_revision = 0
        self._image_revision = 0
        self._left_image_revision = 0
        self._right_image_revision = 0

        command_qos = rclpy.qos.QoSProfile(
            reliability=(
                rclpy.qos.ReliabilityPolicy.RELIABLE
            ),
            history=(
                rclpy.qos.HistoryPolicy.KEEP_LAST
            ),
            depth=1,
        )

        self._cmd_vel_publisher = (
            self.create_publisher(
                Twist,
                cmd_vel_topic,
                command_qos,
            )
        )

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

        self._image_subscription = (
            self.create_subscription(
                Image,
                camera_topic,
                self._image_callback,
                qos_profile_sensor_data,
            )
        )

        self._left_image_subscription = (
            self.create_subscription(
                Image,
                left_camera_topic,
                self._left_image_callback,
                qos_profile_sensor_data,
            )
        )

        self._right_image_subscription = (
            self.create_subscription(
                Image,
                right_camera_topic,
                self._right_image_callback,
                qos_profile_sensor_data,
            )
        )

        self._panorama_timer = (
            self.create_timer(
                panorama_control_period_sec,
                self._panorama_timer_callback,
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
    def has_image(self) -> bool:
        """Return whether at least one camera image was processed."""
        return self._latest_image is not None

    @property
    def latest_image(self):
        """Return the latest camera image in OpenCV BGR format."""
        return self._latest_image

    @property
    def image_revision(self) -> int:
        """Return the number of processed front-camera images."""
        return self._image_revision

    @property
    def latest_front_image(self):
        """Return the latest front-camera BGR image."""
        return self._latest_image

    @property
    def latest_left_image(self):
        """Return the latest left-camera BGR image."""
        return self._latest_left_image

    @property
    def latest_right_image(self):
        """Return the latest right-camera BGR image."""
        return self._latest_right_image

    @property
    def front_image_revision(self) -> int:
        """Return the number of processed front-camera images."""
        return self._image_revision

    @property
    def left_image_revision(self) -> int:
        """Return the number of processed left-camera images."""
        return self._left_image_revision

    @property
    def right_image_revision(self) -> int:
        """Return the number of processed right-camera images."""
        return self._right_image_revision

    @property
    def camera_batch_ready(self) -> bool:
        """Return whether all three camera streams have supplied an image."""
        return (
            self._latest_image is not None
            and self._latest_left_image is not None
            and self._latest_right_image is not None
        )

    @property
    def panorama_acquisition_state(self):
        """Return the current panorama acquisition state, if any."""
        if self._panorama_coordinator is None:
            return None

        return self._panorama_coordinator.state

    @property
    def has_visual_observation(self) -> bool:
        """Return whether the latest panorama has been visually processed."""
        return (
            self._latest_visual_observation
            is not None
        )

    @property
    def latest_visual_observation(self):
        """Return the latest panoramic visual observation result."""
        return self._latest_visual_observation

    @property
    def visual_observation_id(self):
        """Return the latest discrete visual observation ID."""
        if self._latest_visual_observation is None:
            return None

        return (
            self._latest_visual_observation.observation_id
        )

    @property
    def has_place_observation(self) -> bool:
        """Return whether the current cycle has a place observation."""
        return (
            self._latest_place_observation_id
            is not None
        )

    @property
    def place_observation_id(self):
        """Return the latest discrete place observation ID."""
        return self._latest_place_observation_id

    @property
    def place_memory_size(self) -> int:
        """Return the number of stored cognitive places."""
        return len(
            self._place_memory
        )

    @property
    def sensor_snapshot_ready(self) -> bool:
        """Return whether all navigation sensor inputs are available."""
        return (
            self.has_odometry
            and self.has_obstacle_distances
            and self.has_image
        )

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

    def _image_callback(
        self,
        message: Image,
    ) -> None:
        """Convert and store one front-camera image."""
        image = self._image_adapter.to_bgr(
            message
        )

        self._latest_image = image
        self._image_revision += 1

    def _left_image_callback(
        self,
        message: Image,
    ) -> None:
        """Convert and store one left-camera image."""
        image = self._image_adapter.to_bgr(
            message
        )

        self._latest_left_image = image
        self._left_image_revision += 1

    def _right_image_callback(
        self,
        message: Image,
    ) -> None:
        """Convert and store one right-camera image."""
        image = self._image_adapter.to_bgr(
            message
        )

        self._latest_right_image = image
        self._right_image_revision += 1

    def publish_panorama_rotation(
        self,
        goal_yaw_rad: float,
    ):
        """Publish one panorama rotation command toward a yaw target."""
        if self._latest_physical_yaw_rad is None:
            return None

        command = (
            self._panorama_rotation_controller.command(
                current_yaw_rad=(
                    self._latest_physical_yaw_rad
                ),
                goal_yaw_rad=goal_yaw_rad,
            )
        )

        message = (
            self._panorama_motion_adapter.to_twist(
                command
            )
        )

        self._cmd_vel_publisher.publish(
            message
        )

        return command

    def stop_panorama_rotation(
        self,
    ) -> None:
        """Publish an explicit zero-velocity panorama command."""
        message = (
            self._panorama_motion_adapter.stop_twist()
        )

        self._cmd_vel_publisher.publish(
            message
        )

    def capture_camera_batch(
        self,
    ):
        """Freeze the latest front, left, and right camera images."""
        if not self.camera_batch_ready:
            return None

        return (
            self._latest_image.copy(),
            self._latest_left_image.copy(),
            self._latest_right_image.copy(),
        )

    def _camera_revisions(
        self,
    ):
        """Return the current three-camera revision tuple."""
        return (
            self._image_revision,
            self._left_image_revision,
            self._right_image_revision,
        )

    def start_panorama_acquisition(
        self,
    ) -> bool:
        """Start one panorama cycle when yaw and cameras are ready."""
        if (
            self._latest_physical_yaw_rad is None
            or not self.camera_batch_ready
        ):
            return False

        if (
            self._panorama_coordinator is not None
            and not self._panorama_coordinator.is_complete
        ):
            raise RuntimeError(
                "panorama acquisition is already active"
            )

        coordinator = PanoramaCoordinator(
            current_yaw_rad=(
                self._latest_physical_yaw_rad
            ),
            action_count=(
                BaselineMotionSet.ACTION_COUNT
            ),
        )

        coordinator.capture_initial_batch(
            self.capture_camera_batch(),
            self._camera_revisions(),
        )

        self._latest_visual_observation = None
        self._latest_place_observation_id = None
        self._panorama_coordinator = coordinator

        return True

    def step_panorama_acquisition(
        self,
    ):
        """Advance one non-blocking panorama acquisition step."""
        coordinator = self._panorama_coordinator

        if coordinator is None:
            return None

        if coordinator.is_complete:
            return coordinator.state

        if (
            coordinator.state
            is PanoramaCoordinatorState.ROTATING
        ):
            command = self.publish_panorama_rotation(
                coordinator.current_goal_yaw_rad
            )

            if (
                command is not None
                and command.goal_reached
            ):
                coordinator.mark_rotation_reached(
                    self._camera_revisions()
                )

            return coordinator.state

        if (
            coordinator.state
            is PanoramaCoordinatorState.WAIT_FRESH_CAMERAS
        ):
            revisions = self._camera_revisions()

            if coordinator.fresh_camera_batch_ready(
                revisions
            ):
                coordinator.capture_fresh_batch(
                    self.capture_camera_batch(),
                    revisions,
                )

            return coordinator.state

        return coordinator.state

    def completed_panorama_images(
        self,
    ):
        """Return completed panorama images, or None while incomplete."""
        if (
            self._panorama_coordinator is None
            or not self._panorama_coordinator.is_complete
        ):
            return None

        return self._panorama_coordinator.compiled_images()

    def process_completed_visual_observation(
        self,
    ):
        """Convert one completed panorama into a discrete observation."""
        if self._latest_visual_observation is not None:
            return self._latest_visual_observation

        images = self.completed_panorama_images()

        if images is None:
            return None

        result = self._visual_observer.process(
            images
        )

        self._latest_visual_observation = result

        return result

    def resolve_current_place_observation(
        self,
    ):
        """Resolve the current cognitive position to one place ID."""
        if self._latest_place_observation_id is not None:
            return self._latest_place_observation_id

        if (
            self._latest_visual_observation is None
            or self._latest_odometry_state is None
        ):
            return None

        place_id = self._place_memory.resolve_place(
            self._latest_odometry_state.position
        )

        self._latest_place_observation_id = (
            place_id
        )

        return place_id

    def _panorama_timer_callback(
        self,
    ) -> None:
        """Advance an active panorama acquisition without blocking ROS."""
        if (
            self._panorama_coordinator is None
            or self._panorama_coordinator.is_complete
        ):
            return

        self.step_panorama_acquisition()

    def capture_sensor_snapshot(
        self,
    ):
        """Freeze the latest complete navigation sensor state."""
        if not self.sensor_snapshot_ready:
            return None

        return build_sensor_snapshot(
            state=self._latest_odometry_state,
            obstacle_distances=(
                self._latest_obstacle_distances
            ),
            image=self._latest_image,
            odometry_revision=(
                self._odometry_revision
            ),
            scan_revision=(
                self._scan_revision
            ),
            image_revision=(
                self._image_revision
            ),
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
