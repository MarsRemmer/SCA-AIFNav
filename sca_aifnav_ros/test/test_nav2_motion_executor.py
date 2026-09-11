"""Tests for the Nav2 navigation-motion backend."""

from types import SimpleNamespace

import pytest
import rclpy
from action_msgs.msg import GoalStatus
from nav_msgs.msg import Odometry
from rclpy.node import Node

from sca_aifnav_core.planar_geometry import Point2D
from sca_aifnav_ros.navigation_core_bridge import NavigationActionTarget
from sca_aifnav_ros.nav2_motion_executor import Nav2MotionExecutor


class CapturePublisher:
    """Capture published ROS messages."""

    def __init__(self):
        self.messages = []

    def publish(self, message):
        self.messages.append(message)


class ImmediateFuture:
    """Return one deterministic asynchronous result."""

    def __init__(self, result):
        self._result = result

    def result(self):
        return self._result

    def add_done_callback(self, callback):
        callback(self)


class FakeGoalHandle:
    """Represent an accepted deterministic Nav2 goal."""

    def __init__(self, status):
        self.accepted = True
        self.status = status

    def get_result_async(self):
        wrapped = SimpleNamespace(
            status=self.status,
            result=SimpleNamespace(),
        )

        return ImmediateFuture(
            wrapped
        )

    def cancel_goal_async(self):
        return None


class FakeActionClient:
    """Capture NavigateToPose goals."""

    def __init__(self, status=GoalStatus.STATUS_SUCCEEDED):
        self.status = status
        self.goals = []

    def wait_for_server(self, timeout_sec=None):
        return True

    def send_goal_async(
        self,
        goal,
        feedback_callback=None,
    ):
        self.goals.append(goal)

        return ImmediateFuture(
            FakeGoalHandle(
                self.status
            )
        )


@pytest.fixture
def ros_node():
    rclpy.init()

    node = Node(
        "test_nav2_motion_executor"
    )

    yield node

    node.destroy_node()

    if rclpy.ok():
        rclpy.shutdown()


def target(
    *,
    action_id=0,
    x=1.0,
    y=0.0,
    stationary=False,
):
    return NavigationActionTarget(
        action_id=action_id,
        source_place_id=0,
        target_place_id=(
            0 if stationary else 1
        ),
        target_position=Point2D(
            x,
            y,
        ),
        is_stationary=stationary,
    )


def odometry(x=0.0, y=0.0):
    message = Odometry()

    message.pose.pose.position.x = float(x)
    message.pose.pose.position.y = float(y)
    message.pose.pose.orientation.w = 1.0

    return message


def test_stationary_action_does_not_call_nav2(
    ros_node,
):
    action_client = FakeActionClient()
    publisher = CapturePublisher()

    executor = Nav2MotionExecutor(
        node=ros_node,
        action_client=action_client,
        initial_pose_publisher=publisher,
    )

    executor.start(
        target(
            action_id=12,
            stationary=True,
        )
    )

    update = executor.step()

    assert update.completed_action_id == 12
    assert update.failed_action_id is None
    assert executor.is_active is False
    assert action_client.goals == []


def test_directional_target_is_sent_as_map_xy(
    ros_node,
):
    action_client = FakeActionClient()
    publisher = CapturePublisher()

    executor = Nav2MotionExecutor(
        node=ros_node,
        action_client=action_client,
        initial_pose_publisher=publisher,
    )

    executor.update_odometry(
        odometry()
    )

    executor.start(
        target(
            x=1.25,
            y=-0.75,
        )
    )

    assert len(action_client.goals) == 1

    goal = action_client.goals[0]

    assert goal.pose.header.frame_id == "map"
    assert goal.pose.pose.position.x == pytest.approx(
        1.25
    )
    assert goal.pose.pose.position.y == pytest.approx(
        -0.75
    )


def test_nav2_success_completes_action(
    ros_node,
):
    executor = Nav2MotionExecutor(
        node=ros_node,
        action_client=FakeActionClient(
            GoalStatus.STATUS_SUCCEEDED
        ),
        initial_pose_publisher=CapturePublisher(),
    )

    executor.update_odometry(
        odometry()
    )

    executor.start(
        target()
    )

    update = executor.step(
        current_position=Point2D(
            1.0,
            0.0,
        )
    )

    assert update.completed_action_id == 0
    assert update.failed_action_id is None
    assert update.command.goal_reached is True
    assert executor.is_active is False


def test_nav2_abort_far_from_goal_fails_action(
    ros_node,
):
    executor = Nav2MotionExecutor(
        node=ros_node,
        action_client=FakeActionClient(
            GoalStatus.STATUS_ABORTED
        ),
        initial_pose_publisher=CapturePublisher(),
    )

    executor.update_odometry(
        odometry()
    )

    executor.start(
        target(
            x=1.0,
            y=0.0,
        )
    )

    update = executor.step(
        current_position=Point2D(
            0.0,
            0.0,
        )
    )

    assert update.completed_action_id is None
    assert update.failed_action_id == 0
    assert update.command.goal_reached is False


def test_nav2_abort_inside_one_third_radius_is_accepted(
    ros_node,
):
    executor = Nav2MotionExecutor(
        node=ros_node,
        action_client=FakeActionClient(
            GoalStatus.STATUS_ABORTED
        ),
        initial_pose_publisher=CapturePublisher(),
        influence_radius=0.5,
    )

    executor.update_odometry(
        odometry()
    )

    executor.start(
        target(
            x=0.15,
            y=0.15,
        )
    )

    update = executor.step(
        current_position=Point2D(
            0.0,
            0.0,
        )
    )

    assert update.completed_action_id == 0
    assert update.failed_action_id is None
    assert update.command.goal_reached is True


def test_initial_pose_is_published_from_odometry(
    ros_node,
):
    publisher = CapturePublisher()

    executor = Nav2MotionExecutor(
        node=ros_node,
        action_client=FakeActionClient(),
        initial_pose_publisher=publisher,
    )

    executor.update_odometry(
        odometry(
            x=0.4,
            y=-0.2,
        )
    )

    executor.start(
        target()
    )

    assert len(publisher.messages) >= 1

    message = publisher.messages[0]

    assert message.header.frame_id == "map"
    assert (
        message.pose.pose.position.x
        == pytest.approx(0.4)
    )
    assert (
        message.pose.pose.position.y
        == pytest.approx(-0.2)
    )
