"""State management for panoramic visual acquisition."""

from sca_aifnav_ros.panorama_capture import (
    BASELINE_CAMERA_COUNT,
    PanoramaImageAccumulator,
    build_panorama_capture_plan,
)


class PanoramaAcquisitionSession:
    """Coordinate one complete multi-camera panorama acquisition."""

    def __init__(
        self,
        current_yaw_rad: float,
        action_count: int,
        camera_count: int = BASELINE_CAMERA_COUNT,
    ) -> None:
        """Create a panorama acquisition session from the starting yaw."""
        self.plan = build_panorama_capture_plan(
            current_yaw_rad=current_yaw_rad,
            action_count=action_count,
            camera_count=camera_count,
        )

        self._accumulator = (
            PanoramaImageAccumulator(
                camera_count=(
                    self.plan.camera_count
                )
            )
        )

    @property
    def batch_count(
        self,
    ) -> int:
        """Return the number of camera batches already captured."""
        return self._accumulator.batch_count

    @property
    def requires_initial_capture(
        self,
    ) -> bool:
        """Return whether the starting-orientation batch is still missing."""
        return self.batch_count == 0

    @property
    def is_complete(
        self,
    ) -> bool:
        """Return whether all planned camera batches were captured."""
        return (
            self.batch_count
            == self.plan.capture_batch_count
        )

    @property
    def next_goal_yaw_rad(
        self,
    ):
        """
        Return the next physical yaw target.

        No rotation target exists before the initial batch is captured
        or after the panorama acquisition has completed.
        """
        if self.requires_initial_capture:
            return None

        if self.is_complete:
            return None

        goal_index = (
            self.batch_count
            - 1
        )

        return (
            self.plan.absolute_goal_angles[
                goal_index
            ]
        )

    def capture_batch(
        self,
        images,
    ) -> None:
        """Record one simultaneous multi-camera capture."""
        if self.is_complete:
            raise RuntimeError(
                "panorama acquisition is already complete"
            )

        self._accumulator.add_batch(
            images
        )

    def compiled_images(
        self,
    ):
        """Return the complete ordered panorama image sequence."""
        if not self.is_complete:
            raise RuntimeError(
                "panorama acquisition is not complete"
            )

        images = (
            self._accumulator.compiled_images()
        )

        if (
            len(images)
            != self.plan.expected_image_count
        ):
            raise RuntimeError(
                "panorama image count does not match capture plan"
            )

        return images
