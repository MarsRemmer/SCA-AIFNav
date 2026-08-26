"""Tests for panorama reacquisition after visual-processing failure."""

import pytest
import rclpy

from sca_aifnav_ros.navigation_node import (
    NavigationNode,
)
from sca_aifnav_ros.visual_observation import (
    VisualObservationResult,
)


class FakeVisualMemory:
    """Track stitcher resets."""

    def __init__(
        self,
    ):
        self.reset_count = 0

    def reset_stitcher(
        self,
    ):
        """Record one stitcher reset."""
        self.reset_count += 1


class FakeRetryObserver:
    """Return deterministic results for successive panorama attempts."""

    initial_confidence_threshold = 0.9
    confidence_decrement = 0.07
    max_attempts = 5

    def __init__(
        self,
        responses,
    ):
        self.responses = list(
            responses
        )

        self.calls = []

        self.memory = (
            FakeVisualMemory()
        )

    def process_attempt(
        self,
        images,
        confidence_threshold,
        attempt_count,
    ):
        """Process exactly one newly acquired panorama."""
        self.calls.append(
            (
                tuple(images),
                confidence_threshold,
                attempt_count,
            )
        )

        return self.responses.pop(
            0
        )


class CompletePanoramaCoordinator:
    """Provide one already completed panorama."""

    def __init__(
        self,
        images,
    ):
        self.is_complete = True
        self._images = tuple(
            images
        )

    def compiled_images(
        self,
    ):
        """Return this acquisition's images."""
        return self._images


@pytest.fixture
def ros_context():
    """Provide a fresh ROS context."""
    rclpy.init()

    yield

    if rclpy.ok():
        rclpy.shutdown()


def visual_result(
    observation_id=3,
    confidence_threshold=0.83,
    attempt_count=2,
):
    """Create one successful visual observation."""
    return VisualObservationResult(
        observation_id=observation_id,
        match_scores=(1.0,),
        confidence_threshold=(
            confidence_threshold
        ),
        attempt_count=attempt_count,
    )


def test_failed_attempt_reacquires_new_panorama(
    ros_context,
):
    """A failed stitch must cause a completely new acquisition."""
    node = NavigationNode()

    first_images = tuple(
        f"first-{index}"
        for index in range(12)
    )

    second_images = tuple(
        f"second-{index}"
        for index in range(12)
    )

    observer = FakeRetryObserver(
        [
            None,
            visual_result(),
        ]
    )

    node._visual_observer = (
        observer
    )

    node._panorama_coordinator = (
        CompletePanoramaCoordinator(
            first_images
        )
    )

    reacquisitions = []

    def start_new_panorama():
        reacquisitions.append(
            True
        )

        node._panorama_coordinator = (
            CompletePanoramaCoordinator(
                second_images
            )
        )

        return True

    node.start_panorama_acquisition = (
        start_new_panorama
    )

    try:
        first = (
            node.process_completed_visual_observation()
        )

        assert first is None

        assert len(
            reacquisitions
        ) == 1

        second = (
            node.process_completed_visual_observation()
        )

        assert second is not None
        assert second.observation_id == 3

        assert len(
            observer.calls
        ) == 2

        assert (
            observer.calls[0][0]
            == first_images
        )

        assert (
            observer.calls[1][0]
            == second_images
        )

        assert (
            observer.calls[0][1]
            == pytest.approx(0.90)
        )

        assert (
            observer.calls[1][1]
            == pytest.approx(0.83)
        )

        assert (
            observer.calls[0][2]
            == 1
        )

        assert (
            observer.calls[1][2]
            == 2
        )

        assert (
            observer.memory.reset_count
            == 1
        )

    finally:
        node.destroy_node()


def test_five_failed_acquisitions_raise_error(
    ros_context,
):
    """Five failed newly acquired panoramas should terminate the attempt."""
    node = NavigationNode()

    observer = FakeRetryObserver(
        [
            None,
            None,
            None,
            None,
            None,
        ]
    )

    node._visual_observer = (
        observer
    )

    panorama_index = 0

    def panorama_images(
        index,
    ):
        return tuple(
            f"panorama-{index}-{item}"
            for item in range(12)
        )

    node._panorama_coordinator = (
        CompletePanoramaCoordinator(
            panorama_images(0)
        )
    )

    def start_new_panorama():
        nonlocal panorama_index

        panorama_index += 1

        node._panorama_coordinator = (
            CompletePanoramaCoordinator(
                panorama_images(
                    panorama_index
                )
            )
        )

        return True

    node.start_panorama_acquisition = (
        start_new_panorama
    )

    try:
        for _ in range(4):
            assert (
                node.process_completed_visual_observation()
                is None
            )

        with pytest.raises(
            ValueError,
            match="repeated panorama acquisitions",
        ):
            node.process_completed_visual_observation()

        assert len(
            observer.calls
        ) == 5

        thresholds = [
            call[1]
            for call in observer.calls
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

        acquired_batches = [
            call[0]
            for call in observer.calls
        ]

        assert len(
            set(acquired_batches)
        ) == 5

        assert panorama_index == 4

        assert (
            observer.memory.reset_count
            == 1
        )

    finally:
        node.destroy_node()
