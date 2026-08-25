"""Tests for non-blocking panorama acquisition coordination."""

import math

import numpy as np
import pytest

from sca_aifnav_ros.panorama_coordinator import (
    PanoramaCoordinator,
    PanoramaCoordinatorState,
)


def _image(value):
    return np.full(
        (4, 6, 3),
        value,
        dtype=np.uint8,
    )


def _batch(offset):
    return (
        _image(offset),
        _image(offset + 10),
        _image(offset + 20),
    )


def _coordinator():
    return PanoramaCoordinator(
        current_yaw_rad=0.0,
        action_count=13,
        camera_count=3,
    )


def test_initial_state_waits_for_first_capture():
    coordinator = _coordinator()

    assert (
        coordinator.state
        is PanoramaCoordinatorState.WAIT_INITIAL_CAPTURE
    )
    assert coordinator.batch_count == 0
    assert coordinator.current_goal_yaw_rad is None
    assert not coordinator.is_complete


def test_initial_capture_starts_first_rotation():
    coordinator = _coordinator()

    coordinator.capture_initial_batch(
        _batch(1),
        (1, 1, 1),
    )

    assert (
        coordinator.state
        is PanoramaCoordinatorState.ROTATING
    )
    assert coordinator.batch_count == 1
    assert coordinator.current_goal_yaw_rad == pytest.approx(
        math.pi / 4.0
    )


def test_rotation_cannot_be_marked_before_initial_capture():
    coordinator = _coordinator()

    with pytest.raises(
        RuntimeError,
        match="rotation target",
    ):
        coordinator.mark_rotation_reached(
            (1, 1, 1)
        )


def test_reached_rotation_waits_for_fresh_cameras():
    coordinator = _coordinator()

    coordinator.capture_initial_batch(
        _batch(1),
        (1, 1, 1),
    )

    coordinator.mark_rotation_reached(
        (100, 101, 102)
    )

    assert (
        coordinator.state
        is PanoramaCoordinatorState.WAIT_FRESH_CAMERAS
    )
    assert coordinator.current_goal_yaw_rad is None
    assert coordinator.reached_revisions == (
        100,
        101,
        102,
    )


def test_every_camera_must_refresh_after_rotation():
    coordinator = _coordinator()

    coordinator.capture_initial_batch(
        _batch(1),
        (1, 1, 1),
    )

    coordinator.mark_rotation_reached(
        (10, 20, 30)
    )

    assert not coordinator.fresh_camera_batch_ready(
        (11, 20, 31)
    )

    assert not coordinator.fresh_camera_batch_ready(
        (10, 21, 31)
    )

    assert coordinator.fresh_camera_batch_ready(
        (11, 21, 31)
    )


def test_fresh_capture_is_rejected_too_early():
    coordinator = _coordinator()

    coordinator.capture_initial_batch(
        _batch(1),
        (1, 1, 1),
    )

    coordinator.mark_rotation_reached(
        (10, 20, 30)
    )

    with pytest.raises(
        RuntimeError,
        match="not all fresh",
    ):
        coordinator.capture_fresh_batch(
            _batch(2),
            (11, 20, 31),
        )


def test_fresh_capture_advances_to_second_rotation():
    coordinator = _coordinator()

    coordinator.capture_initial_batch(
        _batch(1),
        (1, 1, 1),
    )

    coordinator.mark_rotation_reached(
        (10, 20, 30)
    )

    coordinator.capture_fresh_batch(
        _batch(2),
        (11, 21, 31),
    )

    assert coordinator.batch_count == 2
    assert (
        coordinator.state
        is PanoramaCoordinatorState.ROTATING
    )
    assert coordinator.current_goal_yaw_rad == pytest.approx(
        math.pi / 2.0
    )
    assert coordinator.reached_revisions is None


def test_complete_cycle_produces_twelve_images():
    coordinator = _coordinator()

    coordinator.capture_initial_batch(
        _batch(1),
        (1, 1, 1),
    )

    reached_and_fresh = (
        (
            (100, 100, 100),
            (101, 101, 101),
            _batch(2),
        ),
        (
            (200, 200, 200),
            (201, 201, 201),
            _batch(3),
        ),
        (
            (300, 300, 300),
            (301, 301, 301),
            _batch(4),
        ),
    )

    for (
        reached_revisions,
        fresh_revisions,
        images,
    ) in reached_and_fresh:
        coordinator.mark_rotation_reached(
            reached_revisions
        )

        coordinator.capture_fresh_batch(
            images,
            fresh_revisions,
        )

    assert coordinator.is_complete
    assert (
        coordinator.state
        is PanoramaCoordinatorState.COMPLETE
    )
    assert coordinator.batch_count == 4
    assert coordinator.current_goal_yaw_rad is None

    images = coordinator.compiled_images()

    assert len(images) == 12


def test_completed_images_keep_camera_major_order():
    coordinator = _coordinator()

    batches = (
        _batch(1),
        _batch(2),
        _batch(3),
        _batch(4),
    )

    coordinator.capture_initial_batch(
        batches[0],
        (1, 1, 1),
    )

    for index in range(1, 4):
        reached = index * 100

        coordinator.mark_rotation_reached(
            (
                reached,
                reached,
                reached,
            )
        )

        coordinator.capture_fresh_batch(
            batches[index],
            (
                reached + 1,
                reached + 1,
                reached + 1,
            ),
        )

    images = coordinator.compiled_images()

    expected_values = (
        1,
        2,
        3,
        4,
        11,
        12,
        13,
        14,
        21,
        22,
        23,
        24,
    )

    actual_values = tuple(
        int(image[0, 0, 0])
        for image in images
    )

    assert actual_values == expected_values


def test_compiled_images_rejected_before_completion():
    coordinator = _coordinator()

    with pytest.raises(
        RuntimeError,
        match="not complete",
    ):
        coordinator.compiled_images()


@pytest.mark.parametrize(
    "revisions",
    [
        (1, 2),
        (1, 2, 3, 4),
    ],
)
def test_revision_count_must_match_camera_count(
    revisions,
):
    coordinator = _coordinator()

    with pytest.raises(
        ValueError,
        match="revision count",
    ):
        coordinator.capture_initial_batch(
            _batch(1),
            revisions,
        )


@pytest.mark.parametrize(
    "revisions",
    [
        (True, 1, 1),
        (1.0, 1, 1),
        ("1", 1, 1),
    ],
)
def test_revisions_must_be_integer_values(
    revisions,
):
    coordinator = _coordinator()

    with pytest.raises(
        TypeError,
        match="must be integers",
    ):
        coordinator.capture_initial_batch(
            _batch(1),
            revisions,
        )


def test_revisions_cannot_be_negative():
    coordinator = _coordinator()

    with pytest.raises(
        ValueError,
        match="non-negative",
    ):
        coordinator.capture_initial_batch(
            _batch(1),
            (1, -1, 1),
        )
