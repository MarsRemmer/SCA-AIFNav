"""Cognitive odometry state used by the SCA-AIFNav baseline."""

from dataclasses import dataclass
import math

from sca_aifnav_core.planar_geometry import Point2D


def wrap_heading_2pi(angle_rad: float) -> float:
    """Normalize an angle to the interval [0, 2*pi)."""
    if not math.isfinite(angle_rad):
        raise ValueError("angle_rad must be finite")

    return angle_rad % (2.0 * math.pi)


@dataclass(frozen=True)
class CognitiveOdomState:
    """
    Represent the baseline cognitive odometry state.

    The heading stores the direction of the latest planar displacement,
    rather than the physical robot body yaw.
    """

    position: Point2D
    travel_heading_rad: float

    def __post_init__(self) -> None:
        """Validate and normalize the travel heading."""
        if not math.isfinite(self.travel_heading_rad):
            raise ValueError(
                "travel_heading_rad must be finite"
            )

        object.__setattr__(
            self,
            "travel_heading_rad",
            wrap_heading_2pi(self.travel_heading_rad),
        )

    @property
    def x(self) -> float:
        """Return the x coordinate."""
        return self.position.x

    @property
    def y(self) -> float:
        """Return the y coordinate."""
        return self.position.y

    def as_tuple(self) -> tuple:
        """Return state as an (x, y, heading) tuple."""
        return (
            self.x,
            self.y,
            self.travel_heading_rad,
        )


class BaselineOdomTracker:
    """
    Track cognitive odometry following reference baseline behavior.

    Updating from a position computes the direction of displacement from
    the previous position and stores that direction in [0, 2*pi).
    """

    def __init__(self) -> None:
        """Initialize odometry at the origin."""
        self._state = CognitiveOdomState(
            position=Point2D(0.0, 0.0),
            travel_heading_rad=0.0,
        )

    @property
    def state(self) -> CognitiveOdomState:
        """Return the current cognitive odometry state."""
        return self._state

    def reset(
        self,
        position: Point2D = None,
        travel_heading_rad: float = 0.0,
    ) -> CognitiveOdomState:
        """Reset cognitive odometry to a specified state."""
        if position is None:
            position = Point2D(0.0, 0.0)

        self._state = CognitiveOdomState(
            position=position,
            travel_heading_rad=travel_heading_rad,
        )

        return self._state

    def preview_position(
        self,
        position: Point2D,
    ) -> CognitiveOdomState:
        """
        Compute the state produced by moving to a position.

        The stored tracker state is not modified.
        """
        dx = position.x - self._state.x
        dy = position.y - self._state.y

        travel_heading = wrap_heading_2pi(
            math.atan2(dy, dx)
        )

        return CognitiveOdomState(
            position=position,
            travel_heading_rad=travel_heading,
        )

    def update_position(
        self,
        position: Point2D,
    ) -> CognitiveOdomState:
        """Update odometry from a new planar position."""
        self._state = self.preview_position(position)

        return self._state
