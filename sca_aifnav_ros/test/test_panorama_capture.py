"""Tests for panoramic visual capture planning."""

import math

import numpy as np
import pytest

from sca_aifnav_ros.panorama_capture import (
    PanoramaImageAccumulator,
    TURN_INCREMENT_RAD,
    build_panorama_capture_plan,
)


def coded_image(
    value,
):
    """Create one image carrying a deterministic test value."""
    return np.full(
        (
            2,
            3,
            3,
        ),
        value,
        dtype=np.uint8,
    )


def test_baseline_plan_uses_three_turn_stops():
    """Thirteen actions should yield the twelve-direction capture plan."""
    plan = build_panorama_capture_plan(
        current_yaw_rad=0.0,
        action_count=13,
        camera_count=3,
    )

    assert plan.direction_count == 12
    assert plan.turn_stop_count == 3
    assert plan.capture_batch_count == 4
    assert plan.expected_image_count == 12


def test_relative_goals_use_pi_over_four_spacing():
    """Three-camera capture goals should advance by pi over four."""
    plan = build_panorama_capture_plan(
        current_yaw_rad=0.0,
        action_count=13,
        camera_count=3,
    )

    assert (
        plan.relative_goal_angles
        == pytest.approx(
            (
                TURN_INCREMENT_RAD,
                2.0 * TURN_INCREMENT_RAD,
                3.0 * TURN_INCREMENT_RAD,
            )
        )
    )


def test_absolute_goals_are_relative_to_starting_yaw():
    """Panorama rotations should begin from the physical starting yaw."""
    start_yaw = math.pi / 2.0

    plan = build_panorama_capture_plan(
        current_yaw_rad=start_yaw,
        action_count=13,
        camera_count=3,
    )

    assert (
        plan.absolute_goal_angles
        == pytest.approx(
            (
                3.0 * math.pi / 4.0,
                math.pi,
                5.0 * math.pi / 4.0,
            )
        )
    )


def test_absolute_goals_wrap_at_two_pi():
    """World-frame panorama goals should wrap into one full turn."""
    start_yaw = (
        7.0
        * math.pi
        / 4.0
    )

    plan = build_panorama_capture_plan(
        current_yaw_rad=start_yaw,
        action_count=13,
        camera_count=3,
    )

    assert (
        plan.absolute_goal_angles
        == pytest.approx(
            (
                0.0,
                math.pi / 4.0,
                math.pi / 2.0,
            )
        )
    )


def test_twelve_actions_keep_three_turn_stops():
    """An already-even twelve-action input should remain unchanged."""
    plan = build_panorama_capture_plan(
        current_yaw_rad=0.0,
        action_count=12,
        camera_count=3,
    )

    assert plan.direction_count == 12
    assert plan.turn_stop_count == 3


def test_odd_action_adjustment_is_preserved():
    """Odd action counts should be reduced before stop calculation."""
    plan = build_panorama_capture_plan(
        current_yaw_rad=0.0,
        action_count=11,
        camera_count=3,
    )

    assert plan.direction_count == 10
    assert plan.turn_stop_count == 2
    assert plan.capture_batch_count == 3
    assert plan.expected_image_count == 9


def test_accumulator_compiles_camera_major_order():
    """Final image order should group captures by camera."""
    accumulator = (
        PanoramaImageAccumulator(
            camera_count=3
        )
    )

    accumulator.add_batch(
        (
            coded_image(10),
            coded_image(20),
            coded_image(30),
        )
    )

    accumulator.add_batch(
        (
            coded_image(11),
            coded_image(21),
            coded_image(31),
        )
    )

    accumulator.add_batch(
        (
            coded_image(12),
            coded_image(22),
            coded_image(32),
        )
    )

    accumulator.add_batch(
        (
            coded_image(13),
            coded_image(23),
            coded_image(33),
        )
    )

    compiled = (
        accumulator.compiled_images()
    )

    values = [
        int(
            image[
                0,
                0,
                0,
            ]
        )
        for image in compiled
    ]

    assert values == [
        10,
        11,
        12,
        13,
        20,
        21,
        22,
        23,
        30,
        31,
        32,
        33,
    ]


def test_accumulator_copies_captured_images():
    """Later source-image mutation should not alter stored captures."""
    accumulator = (
        PanoramaImageAccumulator(
            camera_count=3
        )
    )

    front = coded_image(10)

    accumulator.add_batch(
        (
            front,
            coded_image(20),
            coded_image(30),
        )
    )

    front[
        0,
        0,
        0,
    ] = 99

    compiled = (
        accumulator.compiled_images()
    )

    assert (
        compiled[
            0
        ][
            0,
            0,
            0,
        ]
        == 10
    )


def test_accumulator_rejects_wrong_camera_count():
    """Every capture batch should contain one image per camera."""
    accumulator = (
        PanoramaImageAccumulator(
            camera_count=3
        )
    )

    with pytest.raises(
        ValueError,
        match=(
            "image batch must match camera_count"
        ),
    ):
        accumulator.add_batch(
            (
                coded_image(10),
                coded_image(20),
            )
        )


def test_accumulator_rejects_non_array_images():
    """Captured camera values should already be OpenCV arrays."""
    accumulator = (
        PanoramaImageAccumulator(
            camera_count=3
        )
    )

    with pytest.raises(
        TypeError,
        match=(
            "captured images must be numpy arrays"
        ),
    ):
        accumulator.add_batch(
            (
                coded_image(10),
                object(),
                coded_image(30),
            )
        )
