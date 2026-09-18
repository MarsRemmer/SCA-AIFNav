"""Tests for the Nav2 navigation-motion backend."""

from types import SimpleNamespace

import pytest
import rclpy
from action_msgs.msg import GoalStatus
from rclpy.node import Node

from sca_aifnav_core.planar_geometry import Point2D
from sca_aifnav_ros.navigation_core_bridge import NavigationActionTarget
from sca_aifnav_ros.nav2_motion_executor import Nav2MotionExecutor


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


def test_stationary_action_does_not_call_nav2(
    ros_node,
):
    action_client = FakeActionClient()

    executor = Nav2MotionExecutor(
        node=ros_node,
        action_client=action_client,
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


def test_directional_target_is_sent_as_physical_odom_xy(
    ros_node,
):
    """Nav2 should receive the physical target in the odom frame."""
    action_client = FakeActionClient()

    executor = Nav2MotionExecutor(
        node=ros_node,
        action_client=action_client,
    )

    cognitive_target = target(
        x=1.25,
        y=-0.75,
    )

    executor.start(
        cognitive_target,
        physical_target_position=Point2D(
            6.25,
            -2.75,
        ),
    )

    assert len(action_client.goals) == 1

    goal = action_client.goals[0]

    assert goal.pose.header.frame_id == "odom"

    assert (
        goal.pose.pose.position.x
        == pytest.approx(6.25)
    )

    assert (
        goal.pose.pose.position.y
        == pytest.approx(-2.75)
    )

    # The active SCA target remains cognitive.
    assert (
        executor.active_target
        is cognitive_target
    )


def test_directional_target_requires_physical_odom_position(
    ros_node,
):
    """A Nav2 action must not silently treat cognitive XY as odom XY."""
    executor = Nav2MotionExecutor(
        node=ros_node,
        action_client=FakeActionClient(),
    )

    with pytest.raises(
        ValueError,
        match="physical_target_position",
    ):
        executor.start(
            target()
        )


def test_nav2_success_completes_action(
    ros_node,
):
    executor = Nav2MotionExecutor(
        node=ros_node,
        action_client=FakeActionClient(
            GoalStatus.STATUS_SUCCEEDED
        ),
    )

    executor.start(
        target(),
        physical_target_position=Point2D(
            1.0,
            0.0,
        ),
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
    )

    executor.start(
        target(
            x=1.0,
            y=0.0,
        ),
        physical_target_position=Point2D(
            1.0,
            0.0,
        ),
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
        influence_radius=0.5,
    )

    executor.start(
        target(
            x=0.15,
            y=0.15,
        ),
        physical_target_position=Point2D(
            0.15,
            0.15,
        ),
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
