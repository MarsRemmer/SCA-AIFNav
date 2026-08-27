"""Tests for predictive cognitive pose handling during navigation."""

from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
import pytest
import rclpy

from sca_aifnav_core.planar_geometry import (
    Point2D,
)
from sca_aifnav_ros.goal_motion_controller import (
    GoalMotionCommand,
)
from sca_aifnav_ros.navigation_core_bridge import (
    NavigationActionTarget,
)
from sca_aifnav_ros.navigation_motion_executor import (
    NavigationMotionUpdate,
)
from sca_aifnav_ros.navigation_node import (
    NavigationNode,
)
from sca_aifnav_ros.visual_observation import (
    VisualObservationResult,
)


@pytest.fixture
def ros_context():
    """Provide one isolated ROS 2 context."""
    rclpy.init()

    yield

    if rclpy.ok():
        rclpy.shutdown()


def odometry_at(
    x,
    y,
):
    """Create deterministic planar odometry."""
    message = Odometry()

    message.pose.pose.position.x = float(
        x
    )
    message.pose.pose.position.y = float(
        y
    )
    message.pose.pose.orientation.w = 1.0

    return message


def visual_result(
    observation_id=3,
):
    """Create one completed visual observation."""
    return VisualObservationResult(
        observation_id=observation_id,
        match_scores=(1.0,),
        confidence_threshold=0.9,
        attempt_count=1,
    )


def planned_target():
    """Return one deterministic directional cognitive target."""
    return NavigationActionTarget(
        action_id=0,
        source_place_id=0,
        target_place_id=1,
        target_position=Point2D(
            0.65,
            0.0,
        ),
        is_stationary=False,
    )


def prepare_cognitive_places(
    node,
):
    """Create deterministic source and target cognitive places."""
    source_id = node._place_memory.resolve_place(
        Point2D(
            0.0,
            0.0,
        )
    )

    target_id = node._place_memory.resolve_place(
        Point2D(
            0.65,
            0.0,
        )
    )

    assert source_id == 0
    assert target_id == 1

    node._internal_cognitive_place_id = (
        source_id
    )


class FailureMotionExecutor:
    """Return one deterministic failed physical action."""

    def __init__(
        self,
    ):
        self._active_target = None

    @property
    def is_active(
        self,
    ):
        """Return whether a target is active."""
        return self._active_target is not None

    @property
    def active_target(
        self,
    ):
        """Return the active target."""
        return self._active_target

    def start(
        self,
        target,
    ):
        """Store the target being physically executed."""
        self._active_target = target

    def step(
        self,
        current_position=None,
        physical_yaw_rad=None,
        scan=None,
    ):
        """Return one failed update for the current action."""
        action_id = (
            self._active_target.action_id
        )

        return NavigationMotionUpdate(
            action_id=action_id,
            command=GoalMotionCommand(
                linear_speed=0.0,
                angular_speed=0.0,
                distance_to_goal=0.65,
                angular_error_rad=0.0,
                goal_reached=False,
            ),
            completed_action_id=None,
            failed_action_id=action_id,
        )


def test_planned_target_advances_cognitive_pose_before_physical_motion(
    ros_context,
    monkeypatch,
):
    """Planning should advance cognition before physical motion."""
    node = NavigationNode()

    try:
        node._odometry_callback(
            odometry_at(
                1.0,
                2.0,
            )
        )

        prepare_cognitive_places(
            node
        )

        target = planned_target()

        monkeypatch.setattr(
            node._navigation_core_bridge,
            "resolve_planned_action_target",
            lambda: target,
        )

        assert (
            node._latest_odometry_state.position
            == Point2D(
                0.0,
                0.0,
            )
        )

        assert (
            node._internal_cognitive_state.position
            == Point2D(
                0.0,
                0.0,
            )
        )

        started = (
            node.start_planned_navigation_action()
        )

        assert started is True

        # Internal cognition has already predicted arrival at the target.
        assert (
            node._internal_cognitive_state.position
            == target.target_position
        )

        assert (
            node._internal_cognitive_place_id
            == target.target_place_id
        )

        # No new physical odometry has arrived yet.
        assert (
            node._latest_odometry_state.position
            == Point2D(
                0.0,
                0.0,
            )
        )

        assert (
            node._pre_action_cognitive_state.position
            == Point2D(
                0.0,
                0.0,
            )
        )

        assert (
            node._pre_action_cognitive_place_id
            == 0
        )

        assert (
            node.navigation_action_active
            is True
        )

    finally:
        node.destroy_node()


def test_completed_observation_uses_predicted_cognitive_target_not_physical_odom(
    ros_context,
    monkeypatch,
):
    """Use the predicted cognitive target for the next observation."""
    node = NavigationNode()

    try:
        # First raw odometry establishes cognitive origin.
        node._odometry_callback(
            odometry_at(
                1.0,
                2.0,
            )
        )

        prepare_cognitive_places(
            node
        )

        target = planned_target()

        monkeypatch.setattr(
            node._navigation_core_bridge,
            "resolve_planned_action_target",
            lambda: target,
        )

        assert (
            node.start_planned_navigation_action()
            is True
        )

        # Simulate physical execution ending somewhere that is not the
        # exact internally predicted target.
        node._odometry_callback(
            odometry_at(
                1.20,
                2.0,
            )
        )

        assert (
            node._latest_odometry_state.position.x
            == pytest.approx(
                0.20
            )
        )

        assert (
            node._internal_cognitive_state.position.x
            == pytest.approx(
                0.65
            )
        )

        node._latest_obstacle_distances = tuple(
            1.0
            for _ in range(12)
        )

        node._scan_revision = 1

        node._latest_visual_observation = (
            visual_result(
                observation_id=7
            )
        )

        observation = (
            node.process_completed_navigation_observation()
        )

        assert observation is not None

        # The model-facing position is the internally predicted pose.
        assert (
            observation.state.position
            == target.target_position
        )

        assert (
            observation.place_observation
            == target.target_place_id
        )

        assert (
            observation.sensory_observation
            == 7
        )

        # Physical odometry did not create a third cognitive place.
        assert (
            node.place_memory_size
            == 2
        )

    finally:
        node.destroy_node()


def test_failed_action_records_failure_before_cognitive_pose_rollback(
    ros_context,
    monkeypatch,
):
    """Record failure evidence before cognitive pose rollback."""
    node = NavigationNode()

    try:
        node._odometry_callback(
            odometry_at(
                1.0,
                2.0,
            )
        )

        prepare_cognitive_places(
            node
        )

        target = planned_target()

        motion_executor = (
            FailureMotionExecutor()
        )

        node._navigation_motion_executor = (
            motion_executor
        )

        monkeypatch.setattr(
            node._navigation_core_bridge,
            "resolve_planned_action_target",
            lambda: target,
        )

        events = []

        def record_failed_action(
            action_id,
        ):
            # At the instant failure evidence is learned, cognition still
            # represents the predicted target.
            events.append(
                (
                    "failure_recorded",
                    node._internal_cognitive_place_id,
                    node._internal_cognitive_state.position,
                )
            )

            assert action_id == target.action_id

            return target

        monkeypatch.setattr(
            node._navigation_core_bridge,
            "record_failed_action",
            record_failed_action,
        )

        assert (
            node.start_planned_navigation_action()
            is True
        )

        assert (
            node._internal_cognitive_place_id
            == 1
        )

        node._latest_scan_message = (
            LaserScan()
        )

        update = (
            node.step_navigation_action()
        )

        assert (
            update.failed_action_id
            == target.action_id
        )

        # Failure was learned while the predicted target was still active.
        assert events == [
            (
                "failure_recorded",
                1,
                Point2D(
                    0.65,
                    0.0,
                ),
            )
        ]

        # Cognition is then restored to the source state.
        assert (
            node._internal_cognitive_place_id
            == 0
        )

        assert (
            node._internal_cognitive_state.position
            == Point2D(
                0.0,
                0.0,
            )
        )

        # Physical recovery has started after cognitive rollback.
        assert (
            node._returning_after_failed_action
            is True
        )

        return_target = (
            node._navigation_motion_executor
            .active_target
        )

        assert (
            return_target.target_place_id
            == 0
        )

        assert (
            return_target.target_position
            == Point2D(
                0.0,
                0.0,
            )
        )

    finally:
        node.destroy_node()
