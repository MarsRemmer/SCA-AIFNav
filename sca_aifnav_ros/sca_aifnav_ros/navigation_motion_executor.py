"""Lifecycle management for one planned physical navigation action."""

from dataclasses import dataclass
import math
import time
from typing import Optional

import numpy as np
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


STUCK_CHECK_PERIOD_SEC = 5.0
STUCK_POSE_TOLERANCE = 0.1
DEFAULT_INFLUENCE_RADIUS = 0.5


@dataclass(frozen=True)
class NavigationMotionUpdate:
    """Describe one physical navigation control update."""

    action_id: int
    command: GoalMotionCommand
    completed_action_id: Optional[int]
    failed_action_id: Optional[int] = None


class NavigationMotionExecutor:
    """Execute one cognitive navigation target until completion or failure."""

    def __init__(
        self,
        controller=None,
        clock=None,
        stuck_check_period_sec: float = (
            STUCK_CHECK_PERIOD_SEC
        ),
        stuck_pose_tolerance: float = (
            STUCK_POSE_TOLERANCE
        ),
        influence_radius: float = (
            DEFAULT_INFLUENCE_RADIUS
        ),
    ) -> None:
        """Create a navigation motion executor."""
        if controller is None:
            controller = GoalMotionController()

        if clock is None:
            clock = time.monotonic

        if not callable(clock):
            raise TypeError(
                "clock must be callable"
            )

        values = (
            stuck_check_period_sec,
            stuck_pose_tolerance,
            influence_radius,
        )

        if not all(
            math.isfinite(value)
            for value in values
        ):
            raise ValueError(
                "motion lifecycle parameters must be finite"
            )

        if stuck_check_period_sec <= 0.0:
            raise ValueError(
                "stuck_check_period_sec must be positive"
            )

        if stuck_pose_tolerance < 0.0:
            raise ValueError(
                "stuck_pose_tolerance must be non-negative"
            )

        if influence_radius <= 0.0:
            raise ValueError(
                "influence_radius must be positive"
            )

        self.controller = controller
        self.clock = clock

        self.stuck_check_period_sec = float(
            stuck_check_period_sec
        )

        self.stuck_pose_tolerance = float(
            stuck_pose_tolerance
        )

        self.influence_radius = float(
            influence_radius
        )

        self._active_target = None

        self._next_stuck_check_time = None

        # Deliberately survives ordinary action completion.
        # The fixed reference keeps its last movement checkpoint
        # between successful actions.
        self._last_stuck_signature = None

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

        self._next_stuck_check_time = (
            float(self.clock())
            + self.stuck_check_period_sec
        )

    def cancel(
        self,
    ) -> None:
        """Cancel the currently active action."""
        self._active_target = None
        self._next_stuck_check_time = None

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
            command = self._stop_command(
                distance_to_goal=0.0,
                goal_reached=True,
            )

            action_id = target.action_id

            self._finish_active_action()

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

        failure_update = (
            self._check_stuck_motion(
                target=target,
                current_position=current_position,
                physical_yaw_rad=(
                    physical_yaw_rad
                ),
            )
        )

        if failure_update is not None:
            return failure_update

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

            self._finish_active_action()

        return NavigationMotionUpdate(
            action_id=target.action_id,
            command=command,
            completed_action_id=(
                completed_action_id
            ),
        )

    def _check_stuck_motion(
        self,
        target,
        current_position,
        physical_yaw_rad,
    ):
        """Run the fixed-reference periodic no-motion check."""
        if self._next_stuck_check_time is None:
            return None

        now = float(
            self.clock()
        )

        if now < self._next_stuck_check_time:
            return None

        signature = (
            self._movement_signature(
                current_position=(
                    current_position
                ),
                physical_yaw_rad=(
                    physical_yaw_rad
                ),
            )
        )

        stuck = (
            self._last_stuck_signature
            is not None
            and np.allclose(
                signature,
                self._last_stuck_signature,
                atol=self.stuck_pose_tolerance,
            )
        )

        while (
            self._next_stuck_check_time
            <= now
        ):
            self._next_stuck_check_time += (
                self.stuck_check_period_sec
            )

        if not stuck:
            self._last_stuck_signature = (
                signature
            )

            return None

        # The fixed reference clears this checkpoint when
        # its motion action is aborted.
        self._last_stuck_signature = None

        accepted = (
            self._failure_within_goal_tolerance(
                current_position=(
                    current_position
                ),
                target_position=(
                    target.target_position
                ),
            )
        )

        distance_to_goal = (
            current_position.distance_to(
                target.target_position
            )
        )

        action_id = target.action_id

        self._finish_active_action()

        if accepted:
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

    @staticmethod
    def _movement_signature(
        current_position: Point2D,
        physical_yaw_rad,
    ):
        """
        Return the movement signature used by the fixed reference.

        The reference expression adds yaw to the two-element XY array
        through NumPy broadcasting rather than constructing an XYZ pose.
        That runtime behavior is intentionally retained here.
        """
        xy = np.array(
            [
                current_position.x,
                current_position.y,
            ],
            dtype=float,
        )

        return (
            xy
            + np.array(
                [
                    float(
                        physical_yaw_rad
                    )
                ],
                dtype=float,
            )
        )

    def _failure_within_goal_tolerance(
        self,
        current_position: Point2D,
        target_position: Point2D,
    ) -> bool:
        """Accept an aborted motion that is sufficiently close to its goal."""
        tolerance = (
            self.influence_radius
            / 3.0
        )

        current = np.array(
            [
                current_position.x,
                current_position.y,
            ],
            dtype=float,
        )

        target = np.array(
            [
                target_position.x,
                target_position.y,
            ],
            dtype=float,
        )

        return bool(
            np.allclose(
                current,
                target,
                atol=tolerance,
            )
        )

    def _finish_active_action(
        self,
    ) -> None:
        """Clear active-action timing while preserving reference history."""
        self._active_target = None
        self._next_stuck_check_time = None

    @staticmethod
    def _stop_command(
        distance_to_goal: float,
        goal_reached: bool,
    ) -> GoalMotionCommand:
        """Return one explicit zero-velocity terminal command."""
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
