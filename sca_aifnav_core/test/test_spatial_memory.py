"""Tests for fixed-radius baseline place memory."""

import math

import pytest

from sca_aifnav_core.planar_geometry import Point2D
from sca_aifnav_core.spatial_memory import BaselinePlaceMemory


def test_memory_starts_empty():
    memory = BaselinePlaceMemory(influence_radius=0.5)

    assert len(memory) == 0
    assert memory.places() == ()


def test_first_position_creates_place_zero():
    memory = BaselinePlaceMemory(influence_radius=0.5)

    place_id = memory.resolve_place(
        Point2D(0.0, 0.0)
    )

    assert place_id == 0
    assert len(memory) == 1
    assert memory.place(0) == Point2D(0.0, 0.0)


def test_position_inside_radius_reuses_existing_place():
    memory = BaselinePlaceMemory(influence_radius=0.5)

    first_id = memory.resolve_place(
        Point2D(0.0, 0.0)
    )
    second_id = memory.resolve_place(
        Point2D(0.3, 0.0)
    )

    assert first_id == 0
    assert second_id == 0
    assert len(memory) == 1


def test_position_outside_radius_creates_new_place():
    memory = BaselinePlaceMemory(influence_radius=0.5)

    memory.resolve_place(
        Point2D(0.0, 0.0)
    )

    place_id = memory.resolve_place(
        Point2D(0.6, 0.0)
    )

    assert place_id == 1
    assert len(memory) == 2


def test_radius_boundary_does_not_match():
    memory = BaselinePlaceMemory(influence_radius=0.5)

    memory.resolve_place(
        Point2D(0.0, 0.0)
    )

    place_id = memory.resolve_place(
        Point2D(0.5, 0.0),
        save_if_missing=False,
    )

    assert place_id == -1
    assert len(memory) == 1


def test_matching_uses_nearest_place_inside_radius():
    memory = BaselinePlaceMemory(influence_radius=1.1)

    memory.resolve_place(
        Point2D(0.0, 0.0)
    )
    memory.resolve_place(
        Point2D(2.0, 0.0)
    )

    place_id = memory.find_match(
        Point2D(1.2, 0.0)
    )

    assert place_id == 1


def test_missing_position_can_be_left_unstored():
    memory = BaselinePlaceMemory(influence_radius=0.5)

    place_id = memory.resolve_place(
        Point2D(5.0, 5.0),
        save_if_missing=False,
    )

    assert place_id == -1
    assert len(memory) == 0


def test_exact_same_position_reuses_place():
    memory = BaselinePlaceMemory(influence_radius=0.5)

    first_id = memory.resolve_place(
        Point2D(1.0, 2.0)
    )
    second_id = memory.resolve_place(
        Point2D(1.0, 2.0)
    )

    assert first_id == second_id == 0
    assert len(memory) == 1


def test_override_radius_can_be_used_for_lookup():
    memory = BaselinePlaceMemory(influence_radius=0.5)

    memory.resolve_place(
        Point2D(0.0, 0.0)
    )

    assert memory.find_match(
        Point2D(0.8, 0.0)
    ) == -1

    assert memory.find_match(
        Point2D(0.8, 0.0),
        influence_radius=1.0,
    ) == 0


def test_place_ids_follow_creation_order():
    memory = BaselinePlaceMemory(influence_radius=0.5)

    first = memory.resolve_place(
        Point2D(0.0, 0.0)
    )
    second = memory.resolve_place(
        Point2D(1.0, 0.0)
    )
    third = memory.resolve_place(
        Point2D(2.0, 0.0)
    )

    assert (first, second, third) == (0, 1, 2)


def test_unknown_place_id_returns_none():
    memory = BaselinePlaceMemory(influence_radius=0.5)

    memory.resolve_place(
        Point2D(0.0, 0.0)
    )

    assert memory.place(10) is None
    assert memory.place(-1) is None


@pytest.mark.parametrize(
    "invalid_radius",
    [
        0.0,
        -0.1,
        math.inf,
        -math.inf,
        math.nan,
    ],
)
def test_invalid_default_radius_is_rejected(
    invalid_radius,
):
    with pytest.raises(ValueError):
        BaselinePlaceMemory(
            influence_radius=invalid_radius
        )


@pytest.mark.parametrize(
    "invalid_radius",
    [
        0.0,
        -1.0,
        math.inf,
        -math.inf,
        math.nan,
    ],
)
def test_invalid_lookup_radius_is_rejected(
    invalid_radius,
):
    memory = BaselinePlaceMemory(
        influence_radius=0.5
    )

    with pytest.raises(ValueError):
        memory.find_match(
            Point2D(0.0, 0.0),
            influence_radius=invalid_radius,
        )
