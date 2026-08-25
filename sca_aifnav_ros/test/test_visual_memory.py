"""Tests for discrete visual observation memory."""

import numpy as np
import pytest

from sca_aifnav_ros.visual_memory import (
    VisualMemory,
)


def solid_image(
    value,
    height=16,
    width=16,
):
    """Create one constant BGR test image."""
    return np.full(
        (
            height,
            width,
            3,
        ),
        value,
        dtype=np.uint8,
    )


def test_default_visual_memory_configuration():
    """Visual memory should preserve baseline class defaults."""
    memory = VisualMemory()

    assert (
        memory.matching_threshold
        == pytest.approx(0.65)
    )

    assert (
        memory.best_confidence_threshold
        == pytest.approx(0.9)
    )


def test_first_panorama_creates_observation_zero():
    """The first valid panorama should create observation zero."""
    memory = VisualMemory(
        matching_threshold=0.7
    )

    observation_id, scores = (
        memory.process_images(
            [
                solid_image(0),
            ]
        )
    )

    assert observation_id == 0
    assert scores == pytest.approx(
        [1.0]
    )

    assert len(
        memory.views
    ) == 1


def test_identical_panorama_reuses_existing_observation():
    """An identical panorama should match the existing observation."""
    memory = VisualMemory(
        matching_threshold=0.7
    )

    image = solid_image(
        80
    )

    first_id, _ = (
        memory.process_images(
            [image]
        )
    )

    second_id, scores = (
        memory.process_images(
            [
                image.copy(),
            ]
        )
    )

    assert first_id == 0
    assert second_id == 0

    assert scores == pytest.approx(
        [1.0]
    )

    assert len(
        memory.views
    ) == 1


def test_dissimilar_panorama_creates_new_observation():
    """A sufficiently different panorama should create a new ID."""
    memory = VisualMemory(
        matching_threshold=0.7
    )

    first_id, _ = (
        memory.process_images(
            [
                solid_image(0),
            ]
        )
    )

    second_id, scores = (
        memory.process_images(
            [
                solid_image(255),
            ]
        )
    )

    assert first_id == 0
    assert second_id == 1

    assert len(
        scores
    ) == 2

    assert scores[-1] == pytest.approx(
        1.0
    )

    assert len(
        memory.views
    ) == 2


def test_threshold_boundary_creates_new_observation():
    """A score equal to the threshold should create a new view."""
    memory = VisualMemory(
        matching_threshold=0.7
    )

    memory.create_view(
        solid_image(0)
    )

    memory.compare_memory_to_panorama = (
        lambda panorama: [0.7]
    )

    observation_id, scores = (
        memory.process_images(
            [
                solid_image(50),
            ]
        )
    )

    assert observation_id == 1

    assert scores == pytest.approx(
        [
            0.7,
            1.0,
        ]
    )

    assert len(
        memory.views
    ) == 2


def test_score_above_threshold_reuses_observation():
    """A score above the threshold should reuse the matched view."""
    memory = VisualMemory(
        matching_threshold=0.7
    )

    memory.create_view(
        solid_image(0)
    )

    memory.compare_memory_to_panorama = (
        lambda panorama: [0.71]
    )

    observation_id, scores = (
        memory.process_images(
            [
                solid_image(100),
            ]
        )
    )

    assert observation_id == 0

    assert scores == pytest.approx(
        [0.71]
    )

    assert len(
        memory.views
    ) == 1


def test_closest_view_uses_stored_threshold():
    """Lookup should preserve the stored-threshold baseline behavior."""
    memory = VisualMemory(
        matching_threshold=0.7
    )

    memory.create_view(
        solid_image(0)
    )

    memory.compare_memory_to_panorama = (
        lambda panorama: [0.6]
    )

    observation_id, scores = (
        memory.get_closest_view_id(
            solid_image(10),
            matching_threshold=0.5,
        )
    )

    assert observation_id is None

    assert scores == pytest.approx(
        [0.6]
    )


def test_wider_matching_panorama_replaces_memory():
    """A wider matched panorama should replace the stored image."""
    memory = VisualMemory(
        matching_threshold=0.7
    )

    memory.create_view(
        solid_image(
            0,
            width=16,
        )
    )

    memory.compare_memory_to_panorama = (
        lambda panorama: [0.9]
    )

    wider_image = solid_image(
        20,
        width=32,
    )

    observation_id, _ = (
        memory.process_images(
            [
                wider_image,
            ]
        )
    )

    assert observation_id == 0

    assert (
        memory.views[
            0
        ].panorama.shape[1]
        == 32
    )


def test_narrower_matching_panorama_does_not_replace_memory():
    """A narrower matched panorama should leave memory unchanged."""
    memory = VisualMemory(
        matching_threshold=0.7
    )

    original_image = solid_image(
        0,
        width=32,
    )

    memory.create_view(
        original_image
    )

    memory.compare_memory_to_panorama = (
        lambda panorama: [0.9]
    )

    observation_id, _ = (
        memory.process_images(
            [
                solid_image(
                    20,
                    width=16,
                ),
            ]
        )
    )

    assert observation_id == 0

    assert (
        memory.views[
            0
        ].panorama.shape[1]
        == 32
    )


def test_ssim_similarity_is_rounded_to_two_decimals():
    """Visual similarity should be rounded to two decimal places."""
    image = solid_image(
        100
    )

    score = (
        VisualMemory.compare_images_ssim(
            image,
            image.copy(),
        )
    )

    assert score == pytest.approx(
        1.0
    )

    assert score == round(
        score,
        2,
    )


def test_single_large_image_is_downscaled():
    """A single very large panorama should use baseline downscaling."""
    memory = VisualMemory()

    image = solid_image(
        0,
        height=1001,
        width=2001,
    )

    panorama = memory.stitch_images(
        [image]
    )

    assert panorama.shape == (
        250,
        500,
        3,
    )


def test_failed_panorama_returns_no_observation():
    """A failed panorama should not create a visual observation."""
    memory = VisualMemory()

    memory.stitch_images = (
        lambda images,
        confidence_threshold=0.9: None
    )

    observation_id, scores = (
        memory.process_images(
            [
                solid_image(0),
            ]
        )
    )

    assert observation_id is None
    assert scores == []

    assert len(
        memory.views
    ) == 0
