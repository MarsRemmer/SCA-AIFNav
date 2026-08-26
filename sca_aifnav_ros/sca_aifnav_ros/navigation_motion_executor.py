"""Lifecycle management for one planned physical navigation action."""

from dataclasses import dataclass
from typing import Optional

from sensor_msgs.msg import LaserScan

from sca_aifnav_core.planar_geometry import (
    Point2D,
)
from sca_aifnav_ros.goal_motion_controller import (
    GoalMotionCommand,
    GoalMotionController,
)
from sca_aifnav_ros.navigation_core_bridge import (
    NavigationActionTarget,
)


@dataclass(frozen=True)
class NavigationMotionUpdate:
    """Describe one physical navigation control update."""

    action_id: int
    command: GoalMotionCommand
    completed_action_id: Optional[int]


class NavigationMotionExecutor:
    """Execute one cognitive navigation target until completion."""

    def __init__(
        self,
        controller=None,
    ) -> None:
        """Create a navigation motion executor."""
        if controller is None:
            controller = GoalMotionController()

        self.controller = controller
        self._active_target = None

    @property
    def is_active(
        self,
    ) -> bool:
        """Return whether one action is currently being executed."""
        return self._active_target is not None

    @property
    def active_target(
        self,
    ):
        """Return the currently active navigation target."""
        return self._active_target

    def start(
        self,
        target: NavigationActionTarget,
    ) -> None:
        """Start executing one planned navigation target."""
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

    def cancel(
        self,
    ) -> None:
        """Cancel the currently active action."""
        self._active_target = None

    def step(
        self,
        current_position=None,
        physical_yaw_rad=None,
        scan=None,
    ):
        """Advance the active physical action by one control update."""
        target = self._active_target

        if target is None:
            return None

        if target.is_stationary:
            command = GoalMotionCommand(
                linear_speed=0.0,
                angular_speed=0.0,
                distance_to_goal=0.0,
                angular_error_rad=0.0,
                goal_reached=True,
            )

            action_id = target.action_id
            self._active_target = None

            return NavigationMotionUpdate(
                action_id=action_id,
                command=command,
                completed_action_id=action_id,
            )

        if not isinstance(
            current_position,
            Point2D,
        ):
            raise TypeError(
                "current_position must be a Point2D"
            )

        if not isinstance(
            scan,
            LaserScan,
        ):
            raise TypeError(
                "scan must be a LaserScan"
            )

        repulsion = (
            self.controller.repulsion_from_scan(
                scan=scan,
                physical_yaw_rad=(
                    physical_yaw_rad
                ),
            )
        )

        command = self.controller.command(
            current_position=(
                current_position
            ),
            physical_yaw_rad=(
                physical_yaw_rad
            ),
            target_position=(
                target.target_position
            ),
            repulsion=repulsion,
        )

        completed_action_id = None

        if command.goal_reached:
            completed_action_id = (
                target.action_id
            )

            self._active_target = None

        return NavigationMotionUpdate(
            action_id=target.action_id,
            command=command,
            completed_action_id=(
                completed_action_id
            ),
        )
