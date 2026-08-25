"""Planar geometry utilities for the SCA-AIFNav baseline."""

from dataclasses import dataclass
import math


def wrap_angle_rad(angle_rad: float) -> float:
    """Normalize an angle to the interval [-pi, pi)."""
    if not math.isfinite(angle_rad):
        raise ValueError("angle_rad must be finite")

    return (angle_rad + math.pi) % (2.0 * math.pi) - math.pi


def wrap_angle_deg(angle_deg: float) -> float:
    """Normalize an angle to the interval [-180, 180)."""
    if not math.isfinite(angle_deg):
        raise ValueError("angle_deg must be finite")

    return (angle_deg + 180.0) % 360.0 - 180.0


@dataclass(frozen=True)
class Point2D:
    """Represent one Cartesian point in the horizontal plane."""

    x: float
    y: float

    def __post_init__(self) -> None:
        """Validate point coordinates."""
        if not math.isfinite(self.x) or not math.isfinite(self.y):
            raise ValueError("point coordinates must be finite")

    def distance_to(self, other: "Point2D") -> float:
        """Return Euclidean distance to another point."""
        return math.hypot(
            other.x - self.x,
            other.y - self.y,
        )

    def bearing_to(self, other: "Point2D") -> float:
        """Return global bearing toward another point in radians."""
        return math.atan2(
            other.y - self.y,
            other.x - self.x,
        )

    def translated(self, dx: float, dy: float) -> "Point2D":
        """Return a point translated by a Cartesian offset."""
        if not math.isfinite(dx) or not math.isfinite(dy):
            raise ValueError("translation must be finite")

        return Point2D(
            x=self.x + dx,
            y=self.y + dy,
        )


@dataclass(frozen=True)
class PlanarPose:
    """Represent robot position and yaw in the horizontal plane."""

    position: Point2D
    yaw_rad: float

    def __post_init__(self) -> None:
        """Validate and normalize the yaw angle."""
        if not math.isfinite(self.yaw_rad):
            raise ValueError("yaw_rad must be finite")

        object.__setattr__(
            self,
            "yaw_rad",
            wrap_angle_rad(self.yaw_rad),
        )

    @classmethod
    def from_xy_yaw(
        cls,
        x: float,
        y: float,
        yaw_rad: float,
    ) -> "PlanarPose":
        """Construct a planar pose from scalar coordinates."""
        return cls(
            position=Point2D(x=x, y=y),
            yaw_rad=yaw_rad,
        )

    @property
    def x(self) -> float:
        """Return the x coordinate."""
        return self.position.x

    @property
    def y(self) -> float:
        """Return the y coordinate."""
        return self.position.y

    def distance_to(self, other: "PlanarPose") -> float:
        """Return translational distance to another pose."""
        return self.position.distance_to(other.position)

    def global_bearing_to(self, target: Point2D) -> float:
        """Return global bearing from this pose to a target point."""
        return self.position.bearing_to(target)

    def relative_bearing_to(self, target: Point2D) -> float:
        """Return target bearing relative to the robot heading."""
        global_bearing = self.global_bearing_to(target)

        return wrap_angle_rad(
            global_bearing - self.yaw_rad
        )

    def moved_local(
        self,
        forward: float,
        lateral: float = 0.0,
    ) -> "PlanarPose":
        """Return a pose translated in its local robot frame."""
        if not math.isfinite(forward) or not math.isfinite(lateral):
            raise ValueError("local displacement must be finite")

        cos_yaw = math.cos(self.yaw_rad)
        sin_yaw = math.sin(self.yaw_rad)

        dx = forward * cos_yaw - lateral * sin_yaw
        dy = forward * sin_yaw + lateral * cos_yaw

        return PlanarPose(
            position=self.position.translated(dx, dy),
            yaw_rad=self.yaw_rad,
        )

    def with_yaw(self, yaw_rad: float) -> "PlanarPose":
        """Return an equivalent position with a new yaw angle."""
        return PlanarPose(
            position=self.position,
            yaw_rad=yaw_rad,
        )
