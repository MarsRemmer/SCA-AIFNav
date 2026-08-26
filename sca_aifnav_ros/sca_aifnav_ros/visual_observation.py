"""Panoramic visual observation processing for navigation."""

from dataclasses import dataclass
import math

from sca_aifnav_ros.visual_memory import (
    VisualMemory,
)


DEFAULT_MATCHING_THRESHOLD = 0.7
DEFAULT_STITCH_CONFIDENCE_THRESHOLD = 0.9
STITCH_CONFIDENCE_DECREMENT = 0.07
MAX_STITCH_ATTEMPTS = 5


@dataclass(frozen=True)
class VisualObservationResult:
    """Result of one completed panoramic visual observation."""

    observation_id: int
    match_scores: tuple
    confidence_threshold: float
    attempt_count: int


class PanoramicVisualObserver:
    """Convert multi-camera panorama images into a visual observation."""

    def __init__(
        self,
        memory=None,
        matching_threshold: float = DEFAULT_MATCHING_THRESHOLD,
        initial_confidence_threshold: float = (
            DEFAULT_STITCH_CONFIDENCE_THRESHOLD
        ),
        confidence_decrement: float = STITCH_CONFIDENCE_DECREMENT,
        max_attempts: int = MAX_STITCH_ATTEMPTS,
    ) -> None:
        """Create a panoramic visual observation processor."""
        if (
            isinstance(max_attempts, bool)
            or not isinstance(max_attempts, int)
        ):
            raise TypeError(
                "max_attempts must be an integer"
            )

        if max_attempts <= 0:
            raise ValueError(
                "max_attempts must be positive"
            )

        initial_confidence_threshold = float(
            initial_confidence_threshold
        )

        confidence_decrement = float(
            confidence_decrement
        )

        if (
            not math.isfinite(
                initial_confidence_threshold
            )
            or initial_confidence_threshold <= 0.0
        ):
            raise ValueError(
                "initial confidence threshold must be positive and finite"
            )

        if (
            not math.isfinite(
                confidence_decrement
            )
            or confidence_decrement < 0.0
        ):
            raise ValueError(
                "confidence decrement must be non-negative and finite"
            )

        final_confidence = (
            initial_confidence_threshold
            - confidence_decrement
            * (max_attempts - 1)
        )

        if final_confidence <= 0.0:
            raise ValueError(
                "retry confidence threshold must remain positive"
            )

        if memory is None:
            memory = VisualMemory(
                matching_threshold=(
                    matching_threshold
                )
            )

        self.memory = memory
        self.initial_confidence_threshold = (
            initial_confidence_threshold
        )
        self.confidence_decrement = (
            confidence_decrement
        )
        self.max_attempts = max_attempts

    def process_attempt(
        self,
        images,
        confidence_threshold: float,
        attempt_count: int,
    ):
        """Process one newly acquired panorama exactly once."""
        images = tuple(images)

        if len(images) == 0:
            raise ValueError(
                "panorama images cannot be empty"
            )

        (
            observation_id,
            match_scores,
        ) = self.memory.process_images(
            images,
            confidence_threshold=(
                confidence_threshold
            ),
        )

        if (
            not isinstance(observation_id, int)
            or isinstance(observation_id, bool)
        ):
            return None

        return VisualObservationResult(
            observation_id=observation_id,
            match_scores=tuple(
                float(score)
                for score in match_scores
            ),
            confidence_threshold=(
                confidence_threshold
            ),
            attempt_count=attempt_count,
        )

    def process(
        self,
        images,
    ) -> VisualObservationResult:
        """Process one complete panorama image sequence with retries."""
        images = tuple(
            images
        )

        if len(images) == 0:
            raise ValueError(
                "panorama images cannot be empty"
            )

        confidence_threshold = (
            self.initial_confidence_threshold
        )

        for attempt_count in range(
            1,
            self.max_attempts + 1,
        ):
            (
                observation_id,
                match_scores,
            ) = self.memory.process_images(
                images,
                confidence_threshold=(
                    confidence_threshold
                ),
            )

            if (
                isinstance(
                    observation_id,
                    int,
                )
                and not isinstance(
                    observation_id,
                    bool,
                )
            ):
                return VisualObservationResult(
                    observation_id=(
                        observation_id
                    ),
                    match_scores=tuple(
                        float(score)
                        for score in match_scores
                    ),
                    confidence_threshold=(
                        confidence_threshold
                    ),
                    attempt_count=(
                        attempt_count
                    ),
                )

            if attempt_count < self.max_attempts:
                self.memory.reset_stitcher()

                confidence_threshold -= (
                    self.confidence_decrement
                )

        raise ValueError(
            "unable to create a visual observation "
            "from panorama images"
        )
