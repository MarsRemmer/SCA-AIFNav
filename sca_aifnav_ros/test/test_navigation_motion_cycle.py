"""Tests for physical navigation motion in NavigationNode."""

import math

from geometry_msgs.msg import Twist
import pytest
import rclpy
from sensor_msgs.msg import LaserScan

from sca_aifnav_core.baseline_odometry import (
    CognitiveOdomState,
)
from sca_aifnav_core.planar_geometry import (
    Point2D,
)
from sca_aifnav_ros.navigation_core_bridge import (
    NavigationActionTarget,
)
from sca_aifnav_ros.navigation_node import (
    NavigationNode,
)


class CapturePublisher:
    """Capture published velocity messages."""

    def __init__(
        self,
    ):
        self.messages = []

    def publish(
        self,
        message,
    ):
        """Capture one published message."""
        self.messages.append(
            message
        )


class FakeCoreBridge:
    """Provide a deterministic physical navigation target."""

    def __init__(
        self,
        target=None,
    ):
        self.target = target
        self.executed_actions = []

    def resolve_planned_action_target(
        self,
    ):
        """Return the configured planned target."""
        return self.target

    def record_executed_action(
        self,
        action_id,
    ):
        """Capture completion of one physical action."""
        self.executed_actions.append(
            action_id
        )


@pytest.fixture
def ros_context():
    """Provide a fresh ROS 2 context."""
    rclpy.init()

    yield

    if rclpy.ok():
        rclpy.shutdown()


def scan():
    """Create one scan without valid obstacle returns."""
    message = LaserScan()

    message.angle_min = 0.0
    message.angle_increment = 0.1
    message.range_min = 0.1
    message.range_max = 3.5

    message.ranges = [
        math.inf,
    ]

    return message


def directional_target():
    """Create one directional physical target."""
    return NavigationActionTarget(
        action_id=0,
        source_place_id=0,
        target_place_id=1,
        target_position=Point2D(
            1.0,
            0.0,
        ),
        is_stationary=False,
    )


def stationary_target():
    """Create one stationary physical target."""
    return NavigationActionTarget(
        action_id=12,
        source_place_id=0,
        target_place_id=0,
        target_position=Point2D(
            0.0,
            0.0,
        ),
        is_stationary=True,
    )


def set_pose(
    node,
    x,
    y,
    yaw=0.0,
):
    """Set deterministic cached physical and cognitive pose."""
    node._latest_odometry_state = (
        CognitiveOdomState(
            position=Point2D(
                float(x),
                float(y),
            ),
            travel_heading_rad=0.0,
        )
    )

    node._latest_physical_yaw_rad = float(
        yaw
    )


def initialize_cognitive_origin(
    node,
):
    """Represent the already-established initial cognitive location."""
    state = node._internal_cognitive_tracker.reset(
        position=Point2D(
            0.0,
            0.0,
        ),
        travel_heading_rad=0.0,
    )

    node._internal_cognitive_state = state
    node._internal_cognitive_place_id = 0


def test_no_planned_target_does_not_start_action(
    ros_context,
):
    """Physical execution requires a planned core target."""
    node = NavigationNode()

    node._navigation_core_bridge = (
        FakeCoreBridge()
    )

    try:
        assert (
            node.start_planned_navigation_action()
            is False
        )

        assert (
            node.navigation_action_active
            is False
        )
    finally:
        node.destroy_node()


def test_planned_target_starts_physical_action(
    ros_context,
):
    """A valid core target should start physical execution."""
    node = NavigationNode()

    initialize_cognitive_origin(
        node
    )

    node._navigation_core_bridge = (
        FakeCoreBridge(
            directional_target()
        )
    )

    try:
        assert (
            node.start_planned_navigation_action()
            is True
        )

        assert (
            node.navigation_action_active
            is True
        )
    finally:
        node.destroy_node()


def test_directional_action_waits_for_sensor_state(
    ros_context,
):
    """Directional motion must wait for pose, yaw, and scan."""
    node = NavigationNode()

    initialize_cognitive_origin(
        node
    )

    node._navigation_core_bridge = (
        FakeCoreBridge(
            directional_target()
        )
    )

    node.start_planned_navigation_action()

    try:
        assert (
            node.step_navigation_action()
            is None
        )

        assert (
            node.navigation_action_active
            is True
        )
    finally:
        node.destroy_node()


def test_directional_action_publishes_velocity(
    ros_context,
):
    """A distant target should publish a physical velocity command."""
    node = NavigationNode()

    initialize_cognitive_origin(
        node
    )

    bridge = FakeCoreBridge(
        directional_target()
    )

    publisher = CapturePublisher()

    node._navigation_core_bridge = bridge
    node._cmd_vel_publisher = publisher

    set_pose(
        node,
        x=0.0,
        y=0.0,
        yaw=0.0,
    )

    node._latest_scan_message = scan()

    node.start_planned_navigation_action()

    try:
        update = (
            node.step_navigation_action()
        )

        assert update is not None

        assert (
            update.completed_action_id
            is None
        )

        assert (
            node.navigation_action_active
            is True
        )

        assert bridge.executed_actions == []

        assert len(
            publisher.messages
        ) == 1

        message = publisher.messages[0]

        assert isinstance(
            message,
            Twist,
        )

        assert message.linear.x > 0.0
        assert message.angular.z == 0.0
    finally:
        node.destroy_node()


def test_reached_target_records_action_completion(
    ros_context,
):
    """Core completion must occur only after the target is reached."""
    node = NavigationNode()

    initialize_cognitive_origin(
        node
    )

    bridge = FakeCoreBridge(
        directional_target()
    )

    publisher = CapturePublisher()

    node._navigation_core_bridge = bridge
    node._cmd_vel_publisher = publisher

    set_pose(
        node,
        x=1.0,
        y=0.0,
        yaw=0.0,
    )

    node._latest_scan_message = scan()

    node.start_planned_navigation_action()

    try:
        update = (
            node.step_navigation_action()
        )

        assert (
            update.completed_action_id
            == 0
        )

        assert (
            node.navigation_action_active
            is False
        )

        assert (
            bridge.executed_actions
            == [0]
        )

        message = publisher.messages[-1]

        assert message.linear.x == 0.0
        assert message.angular.z == 0.0
    finally:
        node.destroy_node()


def test_stay_completes_without_sensor_state(
    ros_context,
):
    """STAY should complete and publish zero velocity immediately."""
    node = NavigationNode()

    initialize_cognitive_origin(
        node
    )

    bridge = FakeCoreBridge(
        stationary_target()
    )

    publisher = CapturePublisher()

    node._navigation_core_bridge = bridge
    node._cmd_vel_publisher = publisher

    node.start_planned_navigation_action()

    try:
        update = (
            node.step_navigation_action()
        )

        assert (
            update.completed_action_id
            == 12
        )

        assert (
            bridge.executed_actions
            == [12]
        )

        assert (
            node.navigation_action_active
            is False
        )

        message = publisher.messages[-1]

        assert message.linear.x == 0.0
        assert message.angular.z == 0.0
    finally:
        node.destroy_node()
