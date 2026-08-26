"""Tests for physical navigation failure detection."""

from sensor_msgs.msg import LaserScan

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
    NavigationMotionExecutor,
)


class FakeClock:
    """Provide deterministic monotonic time."""

    def __init__(
        self,
    ):
        self.now = 0.0

    def __call__(
        self,
    ):
        return self.now

    def advance(
        self,
        seconds,
    ):
        self.now += seconds


class MovingController:
    """Return one continuing movement command."""

    def repulsion_from_scan(
        self,
        scan,
        physical_yaw_rad,
    ):
        """Return irrelevant deterministic repulsion."""
        return (
            0.0,
            0.0,
        )

    def command(
        self,
        current_position,
        physical_yaw_rad,
        target_position,
        repulsion,
    ):
        """Return a command that has not reached its goal."""
        return GoalMotionCommand(
            linear_speed=0.1,
            angular_speed=0.0,
            distance_to_goal=(
                current_position.distance_to(
                    target_position
                )
            ),
            angular_error_rad=0.0,
            goal_reached=False,
        )


def directional_target(
    target_position=None,
):
    """Create one directional navigation target."""
    if target_position is None:
        target_position = Point2D(
            1.0,
            0.0,
        )

    return NavigationActionTarget(
        action_id=0,
        source_place_id=0,
        target_place_id=1,
        target_position=target_position,
        is_stationary=False,
    )


def scan():
    """Create one valid scan object for lifecycle tests."""
    return LaserScan()


def step_at_origin(
    executor,
):
    """Advance one control step at a stationary physical pose."""
    return executor.step(
        current_position=Point2D(
            0.0,
            0.0,
        ),
        physical_yaw_rad=0.0,
        scan=scan(),
    )


def test_first_stuck_check_only_records_checkpoint():
    """The first five-second check should not yet abort the action."""
    clock = FakeClock()

    executor = NavigationMotionExecutor(
        controller=MovingController(),
        clock=clock,
    )

    executor.start(
        directional_target()
    )

    clock.advance(
        5.0
    )

    update = step_at_origin(
        executor
    )

    assert update.completed_action_id is None
    assert update.failed_action_id is None
    assert executor.is_active is True


def test_second_unchanged_checkpoint_fails_action():
    """Two unchanged movement checks should terminate as failure."""
    clock = FakeClock()

    executor = NavigationMotionExecutor(
        controller=MovingController(),
        clock=clock,
    )

    executor.start(
        directional_target()
    )

    clock.advance(
        5.0
    )

    first = step_at_origin(
        executor
    )

    assert first.failed_action_id is None

    clock.advance(
        5.0
    )

    second = step_at_origin(
        executor
    )

    assert second.completed_action_id is None
    assert second.failed_action_id == 0
    assert second.command.goal_reached is False
    assert second.command.linear_speed == 0.0
    assert second.command.angular_speed == 0.0
    assert executor.is_active is False


def test_failed_motion_inside_one_third_radius_is_accepted():
    """A failed motion close enough to the target should count as success."""
    clock = FakeClock()

    executor = NavigationMotionExecutor(
        controller=MovingController(),
        clock=clock,
        influence_radius=0.5,
    )

    executor.start(
        directional_target(
            target_position=Point2D(
                0.15,
                0.15,
            )
        )
    )

    clock.advance(
        5.0
    )

    step_at_origin(
        executor
    )

    clock.advance(
        5.0
    )

    update = step_at_origin(
        executor
    )

    assert update.completed_action_id == 0
    assert update.failed_action_id is None
    assert update.command.goal_reached is True
    assert executor.is_active is False
