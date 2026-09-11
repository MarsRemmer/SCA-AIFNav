"""Nav2 execution backend for planned SCA-AIFNav navigation actions."""

from typing import Optional

from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseWithCovarianceStamped
from nav2_msgs.action import NavigateToPose
from nav_msgs.msg import Odometry
from rclpy.action import ActionClient
from rclpy.qos import (
    QoSDurabilityPolicy,
    QoSHistoryPolicy,
    QoSProfile,
    QoSReliabilityPolicy,
)

from sca_aifnav_core.planar_geometry import Point2D
from sca_aifnav_ros.goal_motion_controller import GoalMotionCommand
from sca_aifnav_ros.navigation_core_bridge import NavigationActionTarget
from sca_aifnav_ros.navigation_motion_executor import NavigationMotionUpdate


DEFAULT_INFLUENCE_RADIUS = 0.5


class Nav2MotionExecutor:
    """
    Execute one cognitive navigation target through Nav2.

    This backend intentionally preserves the higher-level AIMAPP/SCA
    navigation lifecycle. It only replaces the low-level potential-field
    velocity controller with Nav2 NavigateToPose.

    Nav2 owns /cmd_vel while a translational navigation action is active.
    Panorama rotation remains controlled separately by SCA-AIFNav.
    """

    publishes_cmd_vel = False

    def __init__(
        self,
        node,
        action_name: str = "/navigate_to_pose",
        influence_radius: float = DEFAULT_INFLUENCE_RADIUS,
        action_client=None,
        initial_pose_publisher=None,
    ) -> None:
        """Create the Nav2 motion executor."""
        if influence_radius <= 0.0:
            raise ValueError(
                "influence_radius must be positive"
            )

        self.node = node
        self.influence_radius = float(
            influence_radius
        )

        if action_client is None:
            action_client = ActionClient(
                node,
                NavigateToPose,
                action_name,
            )

        self._action_client = action_client

        if initial_pose_publisher is None:
            qos_profile = QoSProfile(
                history=QoSHistoryPolicy.KEEP_LAST,
                depth=10,
                reliability=QoSReliabilityPolicy.RELIABLE,
                durability=QoSDurabilityPolicy.VOLATILE,
            )

            initial_pose_publisher = (
                node.create_publisher(
                    PoseWithCovarianceStamped,
                    "/initialpose",
                    qos_profile,
                )
            )

        self._initial_pose_publisher = (
            initial_pose_publisher
        )

        self._active_target: Optional[
            NavigationActionTarget
        ] = None

        self._goal_handle = None
        self._goal_future = None
        self._result_future = None

        self._terminal_status = None
        self._feedback_position = None

        self._latest_odometry = None
        self._initial_pose_published = False

    @property
    def is_active(self) -> bool:
        """Return whether one physical action is active."""
        return self._active_target is not None

    @property
    def active_target(self):
        """Return the active cognitive target."""
        return self._active_target

    def update_odometry(
        self,
        message: Odometry,
    ) -> None:
        """Cache the latest physical odometry for Nav2 initialisation."""
        if not isinstance(
            message,
            Odometry,
        ):
            raise TypeError(
                "message must be an Odometry"
            )

        self._latest_odometry = message

    def start(
        self,
        target: NavigationActionTarget,
    ) -> None:
        """Start one planned physical action."""
        if not isinstance(
            target,
            NavigationActionTarget,
        ):
            raise TypeError(
                "target must be a NavigationActionTarget"
            )

        if self._active_target is not None:
            raise RuntimeError(
                "a navigation action is already active"
            )

        self._active_target = target
        self._goal_handle = None
        self._goal_future = None
        self._result_future = None
        self._terminal_status = None
        self._feedback_position = None

        # STAY is handled locally and never sent to Nav2.
        if target.is_stationary:
            return

        # AIMAPP initialises Nav2 localization from odometry.
        # Publish before the first goal whenever odometry is available.
        if not self._initial_pose_published:
            self._publish_initial_pose()

        if not self._action_client.wait_for_server(
            timeout_sec=5.0
        ):
            self.node.get_logger().error(
                "NavigateToPose action server not available"
            )

            self._terminal_status = "failed"
            return

        goal = NavigateToPose.Goal()

        goal.pose.header.frame_id = "map"
        goal.pose.header.stamp = (
            self.node.get_clock()
            .now()
            .to_msg()
        )

        goal.pose.pose.position.x = float(
            target.target_position.x
        )

        goal.pose.pose.position.y = float(
            target.target_position.y
        )

        # AIMAPP's Nav2 client supplies only the target position.
        # Keep the same position-goal semantics here.
        self.node.get_logger().info(
            "Sending Nav2 goal "
            f"x={target.target_position.x:.3f}, "
            f"y={target.target_position.y:.3f}, "
            f"action={target.action_id}"
        )

        self._goal_future = (
            self._action_client.send_goal_async(
                goal,
                feedback_callback=(
                    self._feedback_callback
                ),
            )
        )

        self._goal_future.add_done_callback(
            self._goal_response_callback
        )

    def step(
        self,
        current_position=None,
        physical_yaw_rad=None,
        scan=None,
    ):
        """
        Advance the asynchronous Nav2 action lifecycle.

        physical_yaw_rad and scan are accepted only to keep the same
        executor-facing interface as the legacy potential-field backend.
        Nav2 itself owns obstacle avoidance and velocity control.
        """
        target = self._active_target

        if target is None:
            return None

        if target.is_stationary:
            action_id = target.action_id

            self._finish_active_action()

            return NavigationMotionUpdate(
                action_id=action_id,
                command=self._stop_command(
                    distance_to_goal=0.0,
                    goal_reached=True,
                ),
                completed_action_id=action_id,
            )

        if self._terminal_status is None:
            return NavigationMotionUpdate(
                action_id=target.action_id,
                command=self._stop_command(
                    distance_to_goal=(
                        self._distance_to_target(
                            current_position
                        )
                    ),
                    goal_reached=False,
                ),
                completed_action_id=None,
            )

        action_id = target.action_id

        succeeded = (
            self._terminal_status
            == "succeeded"
        )

        distance_to_goal = (
            self._distance_to_target(
                current_position
            )
        )

        # AIMAPP accepts an aborted motion if the final physical pose is
        # within one third of the cognitive-node influence radius.
        if not succeeded:
            succeeded = (
                self._failure_within_goal_tolerance(
                    current_position
                )
            )

            if succeeded:
                self.node.get_logger().info(
                    "Nav2 goal accepted within "
                    f"{self.influence_radius / 3.0:.3f} m "
                    "failure tolerance"
                )

        self._finish_active_action()

        if succeeded:
            return NavigationMotionUpdate(
                action_id=action_id,
                command=self._stop_command(
                    distance_to_goal=(
                        distance_to_goal
                    ),
                    goal_reached=True,
                ),
                completed_action_id=action_id,
            )

        return NavigationMotionUpdate(
            action_id=action_id,
            command=self._stop_command(
                distance_to_goal=(
                    distance_to_goal
                ),
                goal_reached=False,
            ),
            completed_action_id=None,
            failed_action_id=action_id,
        )

    def cancel(self) -> None:
        """Cancel and clear the current Nav2 action."""
        if self._goal_handle is not None:
            try:
                self._goal_handle.cancel_goal_async()
            except Exception:
                pass

        self._finish_active_action()

    def _goal_response_callback(
        self,
        future,
    ) -> None:
        """Handle acceptance or rejection of a Nav2 goal."""
        try:
            goal_handle = future.result()
        except Exception as exc:
            self.node.get_logger().error(
                f"Nav2 goal request failed: {exc}"
            )

            self._terminal_status = "failed"
            return

        if not goal_handle.accepted:
            self.node.get_logger().warning(
                "Nav2 goal rejected"
            )

            self._terminal_status = "failed"
            return

        self._goal_handle = goal_handle

        self.node.get_logger().info(
            "Nav2 goal accepted"
        )

        # The AIMAPP Nav2 client refreshes /initialpose around goal
        # acceptance. Preserve that observable behaviour.
        self._publish_initial_pose()

        self._result_future = (
            goal_handle.get_result_async()
        )

        self._result_future.add_done_callback(
            self._result_callback
        )

    def _result_callback(
        self,
        future,
    ) -> None:
        """Convert the Nav2 result status into the baseline lifecycle."""
        try:
            wrapped_result = future.result()
            status = wrapped_result.status
        except Exception as exc:
            self.node.get_logger().error(
                f"Nav2 result failed: {exc}"
            )

            self._terminal_status = "failed"
            return

        if status == GoalStatus.STATUS_SUCCEEDED:
            self._terminal_status = "succeeded"

            self.node.get_logger().info(
                "Nav2 goal succeeded"
            )
        else:
            self._terminal_status = "failed"

            self.node.get_logger().warning(
                f"Nav2 goal ended with status {status}"
            )

    def _feedback_callback(
        self,
        feedback_message,
    ) -> None:
        """Cache Nav2's latest physical feedback pose."""
        feedback = feedback_message.feedback

        self._feedback_position = Point2D(
            float(
                feedback.current_pose
                .pose.position.x
            ),
            float(
                feedback.current_pose
                .pose.position.y
            ),
        )

    def _publish_initial_pose(self) -> bool:
        """Publish AIMAPP-compatible Nav2 initial localisation."""
        odometry = self._latest_odometry

        if odometry is None:
            return False

        message = PoseWithCovarianceStamped()

        message.header.frame_id = "map"
        message.header.stamp = (
            odometry.header.stamp
        )

        message.pose.pose.position.x = float(
            odometry.pose.pose.position.x
        )

        message.pose.pose.position.y = float(
            odometry.pose.pose.position.y
        )

        message.pose.pose.position.z = float(
            odometry.pose.pose.position.z
        )

        message.pose.pose.orientation.x = float(
            odometry.pose.pose.orientation.x
        )

        message.pose.pose.orientation.y = float(
            odometry.pose.pose.orientation.y
        )

        message.pose.pose.orientation.z = float(
            odometry.pose.pose.orientation.z
        )

        message.pose.pose.orientation.w = float(
            odometry.pose.pose.orientation.w
        )

        message.pose.covariance = [
            0.25, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.25, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0,
            0.06853892,
        ]

        self._initial_pose_publisher.publish(
            message
        )

        self._initial_pose_published = True

        return True

    def _failure_within_goal_tolerance(
        self,
        current_position,
    ) -> bool:
        """Apply AIMAPP's one-third-radius failed-goal tolerance."""
        target = self._active_target

        if target is None:
            return False

        # AIMAPP uses the latest Nav2 feedback pose when available.
        position = self._feedback_position

        if position is None:
            position = current_position

        if not isinstance(
            position,
            Point2D,
        ):
            return False

        tolerance = (
            self.influence_radius
            / 3.0
        )

        return (
            abs(
                position.x
                - target.target_position.x
            )
            <= tolerance
            and abs(
                position.y
                - target.target_position.y
            )
            <= tolerance
        )

    def _distance_to_target(
        self,
        current_position,
    ) -> float:
        """Return the best available physical distance to the target."""
        target = self._active_target

        if target is None:
            return 0.0

        position = self._feedback_position

        if position is None:
            position = current_position

        if not isinstance(
            position,
            Point2D,
        ):
            return 0.0

        return position.distance_to(
            target.target_position
        )

    def _finish_active_action(self) -> None:
        """Clear one completed Nav2 action."""
        self._active_target = None
        self._goal_handle = None
        self._goal_future = None
        self._result_future = None
        self._terminal_status = None
        self._feedback_position = None

    @staticmethod
    def _stop_command(
        distance_to_goal: float,
        goal_reached: bool,
    ) -> GoalMotionCommand:
        """Return a zero Twist-equivalent lifecycle command."""
        return GoalMotionCommand(
            linear_speed=0.0,
            angular_speed=0.0,
            distance_to_goal=float(
                distance_to_goal
            ),
            angular_error_rad=0.0,
            goal_reached=bool(
                goal_reached
            ),
        )
