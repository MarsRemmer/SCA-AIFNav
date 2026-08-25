"""Non-blocking coordination for panoramic camera acquisition."""

from enum import Enum

from sca_aifnav_ros.panorama_acquisition import (
    PanoramaAcquisitionSession,
)


class PanoramaCoordinatorState(Enum):
    """Discrete states of one panorama acquisition cycle."""

    WAIT_INITIAL_CAPTURE = "wait_initial_capture"
    ROTATING = "rotating"
    WAIT_FRESH_CAMERAS = "wait_fresh_cameras"
    COMPLETE = "complete"


class PanoramaCoordinator:
    """Coordinate capture, rotation, and fresh-frame gating."""

    def __init__(
        self,
        current_yaw_rad: float,
        action_count: int,
        camera_count: int = 3,
    ) -> None:
        """Create one non-blocking panorama coordination cycle."""
        self._session = PanoramaAcquisitionSession(
            current_yaw_rad=current_yaw_rad,
            action_count=action_count,
            camera_count=camera_count,
        )

        self._camera_count = camera_count
        self._state = (
            PanoramaCoordinatorState.WAIT_INITIAL_CAPTURE
        )
        self._reached_revisions = None

    @property
    def state(
        self,
    ) -> PanoramaCoordinatorState:
        """Return the current coordinator state."""
        return self._state

    @property
    def batch_count(
        self,
    ) -> int:
        """Return the number of captured camera batches."""
        return self._session.batch_count

    @property
    def is_complete(
        self,
    ) -> bool:
        """Return whether panorama acquisition is complete."""
        return (
            self._state
            is PanoramaCoordinatorState.COMPLETE
        )

    @property
    def current_goal_yaw_rad(
        self,
    ):
        """Return the active physical yaw target while rotating."""
        if (
            self._state
            is not PanoramaCoordinatorState.ROTATING
        ):
            return None

        return self._session.next_goal_yaw_rad

    @property
    def reached_revisions(
        self,
    ):
        """Return camera revisions recorded at the last reached goal."""
        return self._reached_revisions

    def capture_initial_batch(
        self,
        images,
        camera_revisions,
    ) -> None:
        """Capture the starting-orientation camera batch."""
        if (
            self._state
            is not PanoramaCoordinatorState.WAIT_INITIAL_CAPTURE
        ):
            raise RuntimeError(
                "initial camera batch is not expected"
            )

        self._validate_revisions(
            camera_revisions
        )

        self._session.capture_batch(
            images
        )

        self._advance_after_capture()

    def mark_rotation_reached(
        self,
        camera_revisions,
    ) -> None:
        """Record camera revisions when the yaw target is reached."""
        if (
            self._state
            is not PanoramaCoordinatorState.ROTATING
        ):
            raise RuntimeError(
                "rotation target is not currently active"
            )

        self._reached_revisions = (
            self._validate_revisions(
                camera_revisions
            )
        )

        self._state = (
            PanoramaCoordinatorState.WAIT_FRESH_CAMERAS
        )

    def fresh_camera_batch_ready(
        self,
        camera_revisions,
    ) -> bool:
        """Return whether every camera has updated after rotation."""
        if (
            self._state
            is not PanoramaCoordinatorState.WAIT_FRESH_CAMERAS
        ):
            return False

        revisions = self._validate_revisions(
            camera_revisions
        )

        return all(
            current_revision
            > reached_revision
            for (
                current_revision,
                reached_revision,
            )
            in zip(
                revisions,
                self._reached_revisions,
            )
        )

    def capture_fresh_batch(
        self,
        images,
        camera_revisions,
    ) -> None:
        """Capture one batch after every camera has refreshed."""
        if (
            self._state
            is not PanoramaCoordinatorState.WAIT_FRESH_CAMERAS
        ):
            raise RuntimeError(
                "fresh camera batch is not expected"
            )

        revisions = self._validate_revisions(
            camera_revisions
        )

        if not self.fresh_camera_batch_ready(
            revisions
        ):
            raise RuntimeError(
                "camera images are not all fresh"
            )

        self._session.capture_batch(
            images
        )

        self._reached_revisions = None

        self._advance_after_capture()

    def compiled_images(
        self,
    ):
        """Return the completed ordered panorama image sequence."""
        if not self.is_complete:
            raise RuntimeError(
                "panorama acquisition is not complete"
            )

        return self._session.compiled_images()

    def _advance_after_capture(
        self,
    ) -> None:
        """Advance to rotation or completion after a capture."""
        if self._session.is_complete:
            self._state = (
                PanoramaCoordinatorState.COMPLETE
            )
            return

        self._state = (
            PanoramaCoordinatorState.ROTATING
        )

    def _validate_revisions(
        self,
        camera_revisions,
    ):
        """Validate and normalize one camera revision tuple."""
        try:
            revisions = tuple(
                camera_revisions
            )
        except TypeError as exc:
            raise TypeError(
                "camera revisions must be iterable"
            ) from exc

        if len(revisions) != self._camera_count:
            raise ValueError(
                "camera revision count does not match "
                "camera count"
            )

        for revision in revisions:
            if (
                isinstance(revision, bool)
                or not isinstance(revision, int)
            ):
                raise TypeError(
                    "camera revisions must be integers"
                )

            if revision < 0:
                raise ValueError(
                    "camera revisions must be non-negative"
                )

        return revisions
