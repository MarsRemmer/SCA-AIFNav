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

        # Rotation progress is intentionally independent from the
        # number of successfully captured batches. AIMAPP may time out
        # at one rotation target, skip that capture, and continue with
        # the following target.
        self._initial_capture_complete = False
        self._rotation_goal_index = 0
        self._skipped_rotation_count = 0

    @property
    def batch_count(
        self,
    ) -> int:
        """Return the number of camera batches already captured."""
        return self._accumulator.batch_count

    @property
    def skipped_rotation_count(
        self,
    ) -> int:
        """Return how many timed-out rotation captures were skipped."""
        return self._skipped_rotation_count

    @property
    def requires_initial_capture(
        self,
    ) -> bool:
        """Return whether the starting-orientation batch is still missing."""
        return not self._initial_capture_complete

    @property
    def is_complete(
        self,
    ) -> bool:
        """Return whether every planned rotation target was processed."""
        return (
            self._initial_capture_complete
            and self._rotation_goal_index
            >= len(
                self.plan.absolute_goal_angles
            )
        )

    @property
    def next_goal_yaw_rad(
        self,
    ):
        """
        Return the next physical yaw target.

        No rotation target exists before the initial batch is captured
        or after all rotation targets have been processed.
        """
        if self.requires_initial_capture:
            return None

        if self.is_complete:
            return None

        return (
            self.plan.absolute_goal_angles[
                self._rotation_goal_index
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

        if not self._initial_capture_complete:
            self._initial_capture_complete = True
            return

        self._rotation_goal_index += 1

    def skip_rotation_goal(
        self,
    ) -> None:
        """Advance past one rotation target without capturing images."""
        if self.requires_initial_capture:
            raise RuntimeError(
                "initial camera batch must be captured "
                "before skipping a rotation goal"
            )

        if self.is_complete:
            raise RuntimeError(
                "panorama acquisition is already complete"
            )

        self._rotation_goal_index += 1
        self._skipped_rotation_count += 1

    def compiled_images(
        self,
    ):
        """Return captured images after every planned target was processed."""
        if not self.is_complete:
            raise RuntimeError(
                "panorama acquisition is not complete"
            )

        images = (
            self._accumulator.compiled_images()
        )

        expected_batches = (
            self.plan.capture_batch_count
            - self._skipped_rotation_count
        )

        expected_image_count = (
            expected_batches
            * self.plan.camera_count
        )

        if (
            len(images)
            != expected_image_count
        ):
            raise RuntimeError(
                "panorama image count does not match "
                "processed capture plan"
            )

        return images
