"""Tests for panoramic visual observation processing."""

import pytest

from sca_aifnav_ros.visual_observation import (
    DEFAULT_MATCHING_THRESHOLD,
    DEFAULT_STITCH_CONFIDENCE_THRESHOLD,
    MAX_STITCH_ATTEMPTS,
    PanoramicVisualObserver,
    STITCH_CONFIDENCE_DECREMENT,
)


class FakeVisualMemory:
    """Provide deterministic visual-memory responses."""

    def __init__(
        self,
        responses,
    ):
        self.responses = list(
            responses
        )
        self.calls = []
        self.reset_count = 0

    def process_images(
        self,
        images,
        confidence_threshold=0.9,
    ):
        """Return the next configured processing response."""
        self.calls.append(
            (
                tuple(images),
                confidence_threshold,
            )
        )

        return self.responses.pop(0)

    def reset_stitcher(
        self,
    ):
        """Record one stitcher reset."""
        self.reset_count += 1


def test_default_configuration_matches_runtime_behavior():
    """Default observer settings should match navigation runtime values."""
    observer = PanoramicVisualObserver()

    assert (
        observer.memory.matching_threshold
        == pytest.approx(
            DEFAULT_MATCHING_THRESHOLD
        )
    )

    assert (
        observer.initial_confidence_threshold
        == pytest.approx(
            DEFAULT_STITCH_CONFIDENCE_THRESHOLD
        )
    )

    assert (
        observer.confidence_decrement
        == pytest.approx(
            STITCH_CONFIDENCE_DECREMENT
        )
    )

    assert (
        observer.max_attempts
        == MAX_STITCH_ATTEMPTS
    )


def test_successful_first_attempt_returns_observation():
    """A successful first stitch should return immediately."""
    memory = FakeVisualMemory(
        [
            (
                3,
                [0.2, 0.4, 0.7, 1.0],
            ),
        ]
    )

    observer = PanoramicVisualObserver(
        memory=memory
    )

    result = observer.process(
        ["a", "b", "c"]
    )

    assert result.observation_id == 3

    assert result.match_scores == (
        0.2,
        0.4,
        0.7,
        1.0,
    )

    assert (
        result.confidence_threshold
        == pytest.approx(0.9)
    )

    assert result.attempt_count == 1
    assert memory.reset_count == 0


def test_failed_stitch_retries_with_lower_confidence():
    """Failed panorama stitching should reduce confidence and retry."""
    memory = FakeVisualMemory(
        [
            (None, []),
            (None, []),
            (2, [0.1, 0.2, 1.0]),
        ]
    )

    observer = PanoramicVisualObserver(
        memory=memory
    )

    result = observer.process(
        ["front", "left", "right"]
    )

    thresholds = [
        call[1]
        for call in memory.calls
    ]

    assert thresholds == pytest.approx(
        [
            0.90,
            0.83,
            0.76,
        ]
    )

    assert memory.reset_count == 2
    assert result.observation_id == 2
    assert result.attempt_count == 3

    assert (
        result.confidence_threshold
        == pytest.approx(0.76)
    )


def test_all_failed_attempts_raise_value_error():
    """Repeated stitch failures should stop after the configured limit."""
    memory = FakeVisualMemory(
        [
            (None, []),
            (None, []),
            (None, []),
            (None, []),
            (None, []),
        ]
    )

    observer = PanoramicVisualObserver(
        memory=memory
    )

    with pytest.raises(
        ValueError,
        match="unable to create",
    ):
        observer.process(
            ["a", "b", "c"]
        )

    thresholds = [
        call[1]
        for call in memory.calls
    ]

    assert thresholds == pytest.approx(
        [
            0.90,
            0.83,
            0.76,
            0.69,
            0.62,
        ]
    )

    assert len(memory.calls) == 5
    assert memory.reset_count == 4


def test_same_image_sequence_is_used_for_every_retry():
    """Retries should operate on the same frozen panorama image set."""
    memory = FakeVisualMemory(
        [
            (None, []),
            (4, [1.0]),
        ]
    )

    observer = PanoramicVisualObserver(
        memory=memory
    )

    images = [
        "image0",
        "image1",
        "image2",
    ]

    observer.process(
        images
    )

    assert (
        memory.calls[0][0]
        == memory.calls[1][0]
        == tuple(images)
    )


def test_empty_image_sequence_is_rejected():
    """A visual observation requires at least one camera image."""
    memory = FakeVisualMemory(
        []
    )

    observer = PanoramicVisualObserver(
        memory=memory
    )

    with pytest.raises(
        ValueError,
        match="cannot be empty",
    ):
        observer.process(
            []
        )

    assert memory.calls == []
