"""Discrete planar motion primitives used by the SCA-AIFNav baseline."""

from dataclasses import dataclass
import math
from typing import Tuple


@dataclass(frozen=True)
class MotionPrimitive:
    """Description of one discrete planar motion primitive."""

    action_id: int
    center_deg: float
    is_stationary: bool = False

    @property
    def center_rad(self) -> float:
        """Return the primitive direction in radians."""
        return math.radians(self.center_deg)


class BaselineMotionSet:
    """
    Represent the thirteen-action reference baseline baseline motion set.

    Actions 0 through 11 divide the full plane into twelve 30-degree
    directional sectors. Action 12 represents a stationary action.
    """

    DIRECTION_COUNT = 12
    STAY_ACTION = 12
    ACTION_COUNT = 13
    SECTOR_WIDTH_DEG = 30.0

    def __init__(self) -> None:
        """Initialize all baseline motion primitives."""
        self._actions: Tuple[MotionPrimitive, ...] = tuple(
            MotionPrimitive(
                action_id=action_id,
                center_deg=(action_id + 0.5) * self.SECTOR_WIDTH_DEG,
            )
            for action_id in range(self.DIRECTION_COUNT)
        ) + (
            MotionPrimitive(
                action_id=self.STAY_ACTION,
                center_deg=0.0,
                is_stationary=True,
            ),
        )

    def __len__(self) -> int:
        """Return the total number of actions."""
        return self.ACTION_COUNT

    def action(self, action_id: int) -> MotionPrimitive:
        """Return one motion primitive by its action identifier."""
        self._validate_action_id(action_id)
        return self._actions[action_id]

    def all_actions(self) -> Tuple[MotionPrimitive, ...]:
        """Return all motion primitives."""
        return self._actions

    def action_for_angle_deg(self, angle_deg: float) -> int:
        """
        Convert a planar direction angle to its discrete action.

        Angles are normalized into the interval [0, 360).

        Examples
        --------
        10 deg maps to action 0.
        40 deg maps to action 1.
        -10 deg maps to action 11.

        """
        if not math.isfinite(angle_deg):
            raise ValueError("angle_deg must be finite")

        normalized_deg = angle_deg % 360.0
        action_id = int(normalized_deg // self.SECTOR_WIDTH_DEG)

        return action_id

    def action_for_angle_rad(self, angle_rad: float) -> int:
        """Convert a direction expressed in radians to a discrete action."""
        if not math.isfinite(angle_rad):
            raise ValueError("angle_rad must be finite")

        return self.action_for_angle_deg(math.degrees(angle_rad))

    def reverse_action(self, action_id: int) -> int:
        """Return the action pointing in the opposite direction."""
        self._validate_action_id(action_id)

        if action_id == self.STAY_ACTION:
            return self.STAY_ACTION

        return (
            action_id + self.DIRECTION_COUNT // 2
        ) % self.DIRECTION_COUNT

    def angular_error_deg(
        self,
        action_id: int,
        angle_deg: float,
    ) -> float:
        """
        Return signed shortest angular error from an action to an angle.

        The result lies in the interval [-180, 180).
        """
        primitive = self.action(action_id)

        if primitive.is_stationary:
            raise ValueError(
                "stationary action has no directional angular error"
            )

        error = (
            angle_deg
            - primitive.center_deg
            + 180.0
        ) % 360.0 - 180.0

        return error

    @classmethod
    def is_directional(cls, action_id: int) -> bool:
        """Return whether an action represents planar movement."""
        cls._validate_action_id(action_id)
        return action_id != cls.STAY_ACTION

    @classmethod
    def _validate_action_id(cls, action_id: int) -> None:
        """Validate an action identifier."""
        if isinstance(action_id, bool) or not isinstance(action_id, int):
            raise TypeError("action_id must be an integer")

        if not 0 <= action_id < cls.ACTION_COUNT:
            raise ValueError(
                f"action_id must be in [0, {cls.ACTION_COUNT - 1}]"
            )
