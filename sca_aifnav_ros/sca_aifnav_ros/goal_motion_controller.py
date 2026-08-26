"""Potential-field-style control toward a planar navigation target."""

from dataclasses import dataclass
import math

from sensor_msgs.msg import LaserScan

from sca_aifnav_core.planar_geometry import (
    Point2D,
)


DEFAULT_MAX_LINEAR_SPEED = 0.3
DEFAULT_MAX_ANGULAR_SPEED = 0.2
DEFAULT_ANGULAR_TOLERANCE_RAD = math.pi / 10.0
DEFAULT_DISTANCE_TOLERANCE = 0.05
DEFAULT_OBSTACLE_TOLERANCE = 0.5

LINEAR_GAIN = 0.5
ANGULAR_GAIN = 0.3
ATTRACTION_STRENGTH = 100.0
TURN_SPEED_THRESHOLD = 0.05
FORWARD_FOV_HALF_RAD = math.pi / 2.0


@dataclass(frozen=True)
class GoalMotionCommand:
    """Describe one low-level navigation control update."""

    linear_speed: float
    angular_speed: float
    distance_to_goal: float
    angular_error_rad: float
    goal_reached: bool


def normalize_angle(
    angle_rad: float,
) -> float:
    """Normalize an angle into [-pi, pi]."""
    if not math.isfinite(angle_rad):
        raise ValueError(
            "angle_rad must be finite"
        )

    while angle_rad > math.pi:
        angle_rad -= 2.0 * math.pi

    while angle_rad < -math.pi:
        angle_rad += 2.0 * math.pi

    return angle_rad


class GoalMotionController:
    """Compute baseline-compatible motion toward a planar target."""

    def __init__(
        self,
        max_linear_speed: float = DEFAULT_MAX_LINEAR_SPEED,
        max_angular_speed: float = DEFAULT_MAX_ANGULAR_SPEED,
        angular_tolerance_rad: float = (
            DEFAULT_ANGULAR_TOLERANCE_RAD
        ),
        distance_tolerance: float = (
            DEFAULT_DISTANCE_TOLERANCE
        ),
        obstacle_tolerance: float = (
            DEFAULT_OBSTACLE_TOLERANCE
        ),
    ) -> None:
        """Create a planar goal motion controller."""
        values = (
            (
                "max_linear_speed",
                max_linear_speed,
            ),
            (
                "max_angular_speed",
                max_angular_speed,
            ),
            (
                "angular_tolerance_rad",
                angular_tolerance_rad,
            ),
            (
                "distance_tolerance",
                distance_tolerance,
            ),
            (
                "obstacle_tolerance",
                obstacle_tolerance,
            ),
        )

        for name, value in values:
            if not math.isfinite(value):
                raise ValueError(
                    f"{name} must be finite"
                )

            if value <= 0.0:
                raise ValueError(
                    f"{name} must be positive"
                )

        self.max_linear_speed = float(
            max_linear_speed
        )

        self.max_angular_speed = float(
            max_angular_speed
        )

        self.angular_tolerance_rad = float(
            angular_tolerance_rad
        )

        self.distance_tolerance = float(
            distance_tolerance
        )

        self.obstacle_tolerance = float(
            obstacle_tolerance
        )

    def repulsion_from_scan(
        self,
        scan: LaserScan,
        physical_yaw_rad: float,
    ):
        """Compute the planar obstacle-repulsion vector."""
        if not isinstance(
            scan,
            LaserScan,
        ):
            raise TypeError(
                "scan must be a LaserScan"
            )

        if not math.isfinite(
            physical_yaw_rad
        ):
            raise ValueError(
                "physical_yaw_rad must be finite"
            )

        x_repulsion = 0.0
        y_repulsion = 0.0
        rejected_count = 0

        ranges = tuple(
            scan.ranges
        )

        for index, distance in enumerate(
            ranges
        ):
            angle = (
                scan.angle_min
                + scan.angle_increment * index
                + physical_yaw_rad
            )

            in_fov = (
                -FORWARD_FOV_HALF_RAD
                <= angle
                <= FORWARD_FOV_HALF_RAD
            )

            valid_range = (
                math.isfinite(distance)
                and scan.range_min
                <= distance
                <= scan.range_max
            )

            if not in_fov or not valid_range:
                rejected_count += 1
                continue

            current_charge = (
                1.0
                / (
                    4.0
                    * math.pi
                    * distance
                    * distance
                )
            )

            x_repulsion -= (
                current_charge
                * math.cos(angle)
                * self.obstacle_tolerance
            )

            y_repulsion -= (
                current_charge
                * math.sin(angle)
                * self.obstacle_tolerance
            )

        if (
            len(ranges) == 0
            or rejected_count >= len(ranges)
        ):
            return (
                0.0001,
                0.000000000001,
            )

        return (
            x_repulsion,
            y_repulsion,
        )

    def command(
        self,
        current_position: Point2D,
        physical_yaw_rad: float,
        target_position: Point2D,
        repulsion=(0.0, 0.0),
    ) -> GoalMotionCommand:
        """Compute one control command toward a target position."""
        if not isinstance(
            current_position,
            Point2D,
        ):
            raise TypeError(
                "current_position must be a Point2D"
            )

        if not isinstance(
            target_position,
            Point2D,
        ):
            raise TypeError(
                "target_position must be a Point2D"
            )

        if not math.isfinite(
            physical_yaw_rad
        ):
            raise ValueError(
                "physical_yaw_rad must be finite"
            )

        try:
            x_repulsion, y_repulsion = (
                repulsion
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "repulsion must contain two values"
            ) from exc

        x_repulsion = float(
            x_repulsion
        )
        y_repulsion = float(
            y_repulsion
        )

        if (
            not math.isfinite(x_repulsion)
            or not math.isfinite(y_repulsion)
        ):
            raise ValueError(
                "repulsion values must be finite"
            )

        delta_x = (
            target_position.x
            - current_position.x
        )

        delta_y = (
            target_position.y
            - current_position.y
        )

        distance = math.hypot(
            delta_x,
            delta_y,
        )

        if distance <= self.distance_tolerance:
            return GoalMotionCommand(
                linear_speed=0.0,
                angular_speed=0.0,
                distance_to_goal=distance,
                angular_error_rad=0.0,
                goal_reached=True,
            )

        attraction_scale = (
            ATTRACTION_STRENGTH
            / (
                4.0
                * math.pi
                * distance
                * distance
            )
        )

        x_attraction = (
            attraction_scale
            * delta_x
        )

        y_attraction = (
            attraction_scale
            * delta_y
        )

        x_final = (
            x_attraction
            + x_repulsion
        )

        y_final = (
            y_attraction
            + y_repulsion
        )

        target_angle = math.atan2(
            y_final,
            x_final,
        )

        angular_error = normalize_angle(
            target_angle
            - physical_yaw_rad
        )

        linear_speed = min(
            LINEAR_GAIN * distance,
            self.max_linear_speed,
        )

        angular_speed = max(
            min(
                ANGULAR_GAIN * angular_error,
                self.max_angular_speed,
            ),
            -self.max_angular_speed,
        )

        if (
            abs(angular_error)
            > self.angular_tolerance_rad
            and abs(angular_speed)
            >= TURN_SPEED_THRESHOLD
        ):
            linear_speed = 0.0

        else:
            angular_speed = 0.0

        return GoalMotionCommand(
            linear_speed=float(
                linear_speed
            ),
            angular_speed=float(
                angular_speed
            ),
            distance_to_goal=(
                distance
            ),
            angular_error_rad=(
                angular_error
            ),
            goal_reached=False,
        )
