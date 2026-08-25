"""Panoramic visual capture planning for SCA-AIFNav."""

from dataclasses import dataclass
import math
from typing import Tuple

import numpy as np


BASELINE_CAMERA_COUNT = 3
TURN_INCREMENT_RAD = math.pi / 4.0


@dataclass(frozen=True)
class PanoramaCapturePlan:
    """Describe one panoramic visual acquisition sequence."""

    direction_count: int
    camera_count: int
    turn_stop_count: int
    relative_goal_angles: Tuple[float, ...]
    absolute_goal_angles: Tuple[float, ...]
    capture_batch_count: int
    expected_image_count: int


def build_panorama_capture_plan(
    current_yaw_rad: float,
    action_count: int,
    camera_count: int = BASELINE_CAMERA_COUNT,
) -> PanoramaCapturePlan:
    """
    Build the baseline panoramic visual acquisition plan.

    Odd action counts are first reduced by one. With three cameras,
    one capture occurs immediately and subsequent captures occur after
    each pi/4 positive rotation.
    """
    if (
        isinstance(action_count, bool)
        or not isinstance(
            action_count,
            int,
        )
    ):
        raise TypeError(
            "action_count must be an integer"
        )

    if action_count <= 0:
        raise ValueError(
            "action_count must be positive"
        )

    if (
        isinstance(camera_count, bool)
        or not isinstance(
            camera_count,
            int,
        )
    ):
        raise TypeError(
            "camera_count must be an integer"
        )

    if camera_count <= 0:
        raise ValueError(
            "camera_count must be positive"
        )

    if not math.isfinite(
        current_yaw_rad
    ):
        raise ValueError(
            "current_yaw_rad must be finite"
        )

    direction_count = action_count

    if direction_count % 2 != 0:
        direction_count -= 1

    if camera_count == 1:
        turn_stop_count = (
            direction_count
        )
    elif camera_count == 2:
        turn_stop_count = int(
            direction_count / 2
        )
    elif camera_count == 3:
        turn_stop_count = int(
            direction_count / 4
        )
    else:
        turn_stop_count = 1

    if turn_stop_count <= 0:
        turn_stop_count = 1

    relative_goal_angles = tuple(
        TURN_INCREMENT_RAD
        * float(index + 1)
        for index in range(
            turn_stop_count
        )
    )

    normalized_yaw = (
        current_yaw_rad
        % (2.0 * math.pi)
    )

    absolute_goal_angles = tuple(
        (
            normalized_yaw
            + relative_angle
        )
        % (2.0 * math.pi)
        for relative_angle
        in relative_goal_angles
    )

    capture_batch_count = (
        turn_stop_count
        + 1
    )

    expected_image_count = (
        capture_batch_count
        * camera_count
    )

    return PanoramaCapturePlan(
        direction_count=direction_count,
        camera_count=camera_count,
        turn_stop_count=turn_stop_count,
        relative_goal_angles=(
            relative_goal_angles
        ),
        absolute_goal_angles=(
            absolute_goal_angles
        ),
        capture_batch_count=(
            capture_batch_count
        ),
        expected_image_count=(
            expected_image_count
        ),
    )


class PanoramaImageAccumulator:
    """Collect simultaneous camera captures for one panorama."""

    def __init__(
        self,
        camera_count: int = BASELINE_CAMERA_COUNT,
    ) -> None:
        """Initialize an empty panorama capture buffer."""
        if (
            isinstance(camera_count, bool)
            or not isinstance(
                camera_count,
                int,
            )
        ):
            raise TypeError(
                "camera_count must be an integer"
            )

        if camera_count <= 0:
            raise ValueError(
                "camera_count must be positive"
            )

        self.camera_count = camera_count
        self._batches = []

    @property
    def batch_count(
        self,
    ) -> int:
        """Return the number of captured orientation batches."""
        return len(
            self._batches
        )

    def add_batch(
        self,
        images,
    ) -> None:
        """Store one simultaneous multi-camera capture."""
        images = tuple(
            images
        )

        if (
            len(images)
            != self.camera_count
        ):
            raise ValueError(
                "image batch must match camera_count"
            )

        frozen_images = []

        for image in images:
            if not isinstance(
                image,
                np.ndarray,
            ):
                raise TypeError(
                    "captured images must be numpy arrays"
                )

            frozen_images.append(
                image.copy()
            )

        self._batches.append(
            tuple(
                frozen_images
            )
        )

    def compiled_images(
        self,
    ):
        """
        Return images in baseline camera-major ordering.

        All captures from camera zero are returned first, followed by
        all captures from camera one, and so on.
        """
        compiled = []

        for camera_index in range(
            self.camera_count
        ):
            for batch in self._batches:
                compiled.append(
                    batch[
                        camera_index
                    ].copy()
                )

        return tuple(
            compiled
        )
