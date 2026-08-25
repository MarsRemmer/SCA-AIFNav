"""Spatial place memory for the SCA-AIFNav baseline."""

import math
from typing import List, Optional, Tuple

from sca_aifnav_core.planar_geometry import Point2D


class BaselinePlaceMemory:
    """
    Store discrete places using a fixed spatial influence radius.

    A queried position matches the nearest stored place whose Euclidean
    distance is strictly smaller than the influence radius. If no place
    matches, a new place can optionally be created.
    """

    def __init__(self, influence_radius: float = 0.5) -> None:
        """Initialize an empty fixed-radius place memory."""
        if not math.isfinite(influence_radius):
            raise ValueError("influence_radius must be finite")

        if influence_radius <= 0.0:
            raise ValueError("influence_radius must be positive")

        self._influence_radius = float(influence_radius)
        self._places: List[Point2D] = []

    @property
    def influence_radius(self) -> float:
        """Return the baseline fixed influence radius."""
        return self._influence_radius

    def __len__(self) -> int:
        """Return the number of stored places."""
        return len(self._places)

    def places(self) -> Tuple[Point2D, ...]:
        """Return all stored places in ID order."""
        return tuple(self._places)

    def place(self, place_id: int) -> Optional[Point2D]:
        """Return a stored place, or None when its ID does not exist."""
        if isinstance(place_id, bool) or not isinstance(place_id, int):
            raise TypeError("place_id must be an integer")

        if place_id < 0 or place_id >= len(self._places):
            return None

        return self._places[place_id]

    def find_match(
        self,
        position: Point2D,
        influence_radius: Optional[float] = None,
    ) -> int:
        """
        Return the nearest matching place ID.

        A place matches only when its distance is strictly smaller than the
        active influence radius. Return -1 when no place matches.
        """
        radius = self._resolve_radius(influence_radius)

        best_id = -1
        best_distance = radius

        for place_id, stored_position in enumerate(self._places):
            distance = position.distance_to(stored_position)

            if distance < best_distance:
                best_id = place_id
                best_distance = distance

        return best_id

    def resolve_place(
        self,
        position: Point2D,
        save_if_missing: bool = True,
        influence_radius: Optional[float] = None,
    ) -> int:
        """
        Resolve a position to a stored place ID.

        If no stored place matches and save_if_missing is True, append the
        queried position as a new place. Otherwise return -1.
        """
        place_id = self.find_match(
            position=position,
            influence_radius=influence_radius,
        )

        if place_id >= 0:
            return place_id

        if not save_if_missing:
            return -1

        self._places.append(position)

        return len(self._places) - 1

    def _resolve_radius(
        self,
        influence_radius: Optional[float],
    ) -> float:
        """Return and validate the radius used for one lookup."""
        if influence_radius is None:
            return self._influence_radius

        if not math.isfinite(influence_radius):
            raise ValueError("influence_radius must be finite")

        if influence_radius <= 0.0:
            raise ValueError("influence_radius must be positive")

        return float(influence_radius)
