"""Tests for panoramic acquisition state management."""

import math

import numpy as np
import pytest

from sca_aifnav_ros.panorama_acquisition import (
    PanoramaAcquisitionSession,
)


def coded_image(
    value,
):
    """Create one deterministic camera image."""
    return np.full(
        (
            2,
            3,
            3,
        ),
        value,
        dtype=np.uint8,
    )


def camera_batch(
    front,
    left,
    right,
):
    """Create one front-left-right camera capture."""
    return (
        coded_image(front),
        coded_image(left),
        coded_image(right),
    )


def test_session_starts_waiting_for_initial_capture():
    """A new session should capture before requesting rotation."""
    session = PanoramaAcquisitionSession(
        current_yaw_rad=0.0,
        action_count=13,
    )

    assert session.batch_count == 0

    assert (
        session.requires_initial_capture
        is True
    )

    assert session.is_complete is False

    assert (
        session.next_goal_yaw_rad
        is None
    )


def test_initial_capture_exposes_first_rotation_goal():
    """After the initial batch the first yaw target should become active."""
    session = PanoramaAcquisitionSession(
        current_yaw_rad=0.0,
        action_count=13,
    )

    session.capture_batch(
        camera_batch(
            10,
            20,
            30,
        )
    )

    assert session.batch_count == 1

    assert (
        session.requires_initial_capture
        is False
    )

    assert (
        session.next_goal_yaw_rad
        == pytest.approx(
            math.pi / 4.0
        )
    )


def test_successive_captures_advance_rotation_targets():
    """Each captured rotated batch should advance to the next goal."""
    session = PanoramaAcquisitionSession(
        current_yaw_rad=0.0,
        action_count=13,
    )

    session.capture_batch(
        camera_batch(
            10,
            20,
            30,
        )
    )

    assert (
        session.next_goal_yaw_rad
        == pytest.approx(
            math.pi / 4.0
        )
    )

    session.capture_batch(
        camera_batch(
            11,
            21,
            31,
        )
    )

    assert (
        session.next_goal_yaw_rad
        == pytest.approx(
            math.pi / 2.0
        )
    )

    session.capture_batch(
        camera_batch(
            12,
            22,
            32,
        )
    )

    assert (
        session.next_goal_yaw_rad
        == pytest.approx(
            3.0 * math.pi / 4.0
        )
    )


def test_starting_yaw_offsets_all_rotation_targets():
    """Rotation goals should remain relative to the starting body yaw."""
    session = PanoramaAcquisitionSession(
        current_yaw_rad=math.pi / 2.0,
        action_count=13,
    )

    session.capture_batch(
        camera_batch(
            10,
            20,
            30,
        )
    )

    assert (
        session.next_goal_yaw_rad
        == pytest.approx(
            3.0 * math.pi / 4.0
        )
    )

    session.capture_batch(
        camera_batch(
            11,
            21,
            31,
        )
    )

    assert (
        session.next_goal_yaw_rad
        == pytest.approx(
            math.pi
        )
    )


def test_rotation_targets_wrap_at_two_pi():
    """Panorama yaw targets should wrap into one positive revolution."""
    session = PanoramaAcquisitionSession(
        current_yaw_rad=(
            7.0
            * math.pi
            / 4.0
        ),
        action_count=13,
    )

    session.capture_batch(
        camera_batch(
            10,
            20,
            30,
        )
    )

    assert (
        session.next_goal_yaw_rad
        == pytest.approx(0.0)
    )

    session.capture_batch(
        camera_batch(
            11,
            21,
            31,
        )
    )

    assert (
        session.next_goal_yaw_rad
        == pytest.approx(
            math.pi / 4.0
        )
    )


def test_four_batches_complete_baseline_panorama():
    """Three-camera baseline acquisition should finish after four batches."""
    session = PanoramaAcquisitionSession(
        current_yaw_rad=0.0,
        action_count=13,
    )

    session.capture_batch(
        camera_batch(
            10,
            20,
            30,
        )
    )
    session.capture_batch(
        camera_batch(
            11,
            21,
            31,
        )
    )
    session.capture_batch(
        camera_batch(
            12,
            22,
            32,
        )
    )
    session.capture_batch(
        camera_batch(
            13,
            23,
            33,
        )
    )

    assert session.batch_count == 4
    assert session.is_complete is True

    assert (
        session.next_goal_yaw_rad
        is None
    )


def test_complete_session_compiles_twelve_images():
    """A finished acquisition should produce twelve ordered images."""
    session = PanoramaAcquisitionSession(
        current_yaw_rad=0.0,
        action_count=13,
    )

    session.capture_batch(
        camera_batch(
            10,
            20,
            30,
        )
    )
    session.capture_batch(
        camera_batch(
            11,
            21,
            31,
        )
    )
    session.capture_batch(
        camera_batch(
            12,
            22,
            32,
        )
    )
    session.capture_batch(
        camera_batch(
            13,
            23,
            33,
        )
    )

    images = (
        session.compiled_images()
    )

    values = [
        int(
            image[
                0,
                0,
                0,
            ]
        )
        for image in images
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


def test_images_cannot_be_compiled_before_completion():
    """Incomplete acquisition should not masquerade as a panorama."""
    session = PanoramaAcquisitionSession(
        current_yaw_rad=0.0,
        action_count=13,
    )

    session.capture_batch(
        camera_batch(
            10,
            20,
            30,
        )
    )

    with pytest.raises(
        RuntimeError,
        match=(
            "panorama acquisition is not complete"
        ),
    ):
        session.compiled_images()


def test_extra_capture_is_rejected_after_completion():
    """A completed acquisition should reject additional camera batches."""
    session = PanoramaAcquisitionSession(
        current_yaw_rad=0.0,
        action_count=13,
    )

    for offset in range(
        4
    ):
        session.capture_batch(
            camera_batch(
                10 + offset,
                20 + offset,
                30 + offset,
            )
        )

    with pytest.raises(
        RuntimeError,
        match=(
            "panorama acquisition is already complete"
        ),
    ):
        session.capture_batch(
            camera_batch(
                99,
                99,
                99,
            )
        )


def test_invalid_camera_batch_is_rejected():
    """Session capture should preserve accumulator camera-count validation."""
    session = PanoramaAcquisitionSession(
        current_yaw_rad=0.0,
        action_count=13,
    )

    with pytest.raises(
        ValueError,
        match=(
            "image batch must match camera_count"
        ),
    ):
        session.capture_batch(
            (
                coded_image(10),
                coded_image(20),
            )
        )
