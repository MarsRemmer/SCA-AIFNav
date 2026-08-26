"""Panorama rotation control mathematics for SCA-AIFNav."""

from dataclasses import dataclass
import math


DEFAULT_MAX_ANGULAR_SPEED = 1.0
ANGULAR_SPEED_GAIN = 0.6
GOAL_TOLERANCE_RAD = 0.1
WRAP_GOAL_THRESHOLD_RAD = 0.1
WRAP_CURRENT_THRESHOLD_RAD = 5.4


@dataclass(frozen=True)
class PanoramaRotationCommand:
    """Describe one panorama rotation control update."""

    requested_goal_yaw_rad: float
    effective_goal_yaw_rad: float
    current_yaw_rad: float
    angular_error_rad: float
    angular_speed_rad_s: float
    goal_reached: bool


class PanoramaRotationController:
    """Compute baseline-compatible positive panorama rotation commands."""

    def __init__(
        self,
        max_angular_speed: float = (
            DEFAULT_MAX_ANGULAR_SPEED
        ),
        angular_speed_gain: float = (
            ANGULAR_SPEED_GAIN
        ),
        goal_tolerance_rad: float = (
            GOAL_TOLERANCE_RAD
        ),
    ) -> None:
        """Initialize panorama rotation parameters."""
        values = (
            max_angular_speed,
            angular_speed_gain,
            goal_tolerance_rad,
        )

        if not all(
            math.isfinite(value)
            for value in values
        ):
            raise ValueError(
                "rotation parameters must be finite"
            )

        if max_angular_speed <= 0.0:
            raise ValueError(
                "max_angular_speed must be positive"
            )

        if angular_speed_gain <= 0.0:
            raise ValueError(
                "angular_speed_gain must be positive"
            )

        if goal_tolerance_rad < 0.0:
            raise ValueError(
                "goal_tolerance_rad must be non-negative"
            )

        self.max_angular_speed = float(
            max_angular_speed
        )

        self.angular_speed_gain = float(
            angular_speed_gain
        )

        self.goal_tolerance_rad = float(
            goal_tolerance_rad
        )

    @staticmethod
    def effective_goal_yaw(
        current_yaw_rad: float,
        goal_yaw_rad: float,
    ) -> float:
        """
        Return the baseline effective target used near the 2*pi boundary.

        A target close to zero is temporarily represented as 2*pi when
        the robot is already near the end of the positive revolution.
        """
        if not math.isfinite(
            current_yaw_rad
        ):
            raise ValueError(
                "current_yaw_rad must be finite"
            )

        if not math.isfinite(
            goal_yaw_rad
        ):
            raise ValueError(
                "goal_yaw_rad must be finite"
            )

        effective_goal = float(
            goal_yaw_rad
        )

        if (
            effective_goal
            <= WRAP_GOAL_THRESHOLD_RAD
            and current_yaw_rad
            >= WRAP_CURRENT_THRESHOLD_RAD
        ):
            effective_goal = min(
                max(
                    effective_goal
                    + 2.0 * math.pi,
                    WRAP_CURRENT_THRESHOLD_RAD,
                ),
                2.0 * math.pi,
            )

        return effective_goal

    def command(
        self,
        current_yaw_rad: float,
        goal_yaw_rad: float,
    ) -> PanoramaRotationCommand:
        """Compute one baseline panorama rotation control command."""
        effective_goal = (
            self.effective_goal_yaw(
                current_yaw_rad=(
                    current_yaw_rad
                ),
                goal_yaw_rad=(
                    goal_yaw_rad
                ),
            )
        )

        angular_error = abs(
            effective_goal
            - current_yaw_rad
        )

        goal_reached = (
            angular_error
            <= self.goal_tolerance_rad
        )

        if goal_reached:
            angular_speed = 0.0
        else:
            angular_speed = (
                self.angular_speed_gain
                * angular_error
            )

            angular_speed = min(
                angular_speed,
                self.max_angular_speed,
            )

        return PanoramaRotationCommand(
            requested_goal_yaw_rad=float(
                goal_yaw_rad
            ),
            effective_goal_yaw_rad=(
                effective_goal
            ),
            current_yaw_rad=float(
                current_yaw_rad
            ),
            angular_error_rad=(
                angular_error
            ),
            angular_speed_rad_s=(
                angular_speed
            ),
            goal_reached=(
                goal_reached
            ),
        )
