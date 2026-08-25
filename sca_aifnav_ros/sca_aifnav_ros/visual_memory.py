"""Visual observation memory for SCA-AIFNav."""

import cv2
import numpy as np
from skimage.metrics import structural_similarity
from stitching import Stitcher


class VisualView:
    """Store one memorized panoramic visual observation."""

    def __init__(
        self,
        observation_id: int,
        panorama: np.ndarray,
    ) -> None:
        """Create one visual memory entry."""
        self.observation_id = int(
            observation_id
        )
        self.panorama = panorama

    def update(
        self,
        panorama: np.ndarray,
    ) -> None:
        """Replace the stored panorama."""
        self.panorama = panorama


class VisualMemory:
    """
    Match panoramic observations against memorized visual views.

    New observations are created when no memorized panorama exceeds the
    active similarity threshold. Existing views are identified by their
    position in visual memory.
    """

    def __init__(
        self,
        matching_threshold: float = 0.65,
        best_confidence_threshold: float = 0.9,
    ) -> None:
        """Initialize empty visual memory."""
        self.views = []

        self.matching_threshold = float(
            matching_threshold
        )

        self.best_confidence_threshold = float(
            best_confidence_threshold
        )

        self.reset_stitcher()

    def set_memory_views(
        self,
        views,
    ) -> None:
        """Replace the currently memorized visual views."""
        self.views = views

    def get_memory_views(
        self,
    ):
        """Return the currently memorized visual views."""
        return self.views

    def create_view(
        self,
        panorama: np.ndarray,
    ) -> VisualView:
        """Create and store one new visual observation."""
        view = VisualView(
            observation_id=len(
                self.views
            ),
            panorama=panorama,
        )

        self.views.append(
            view
        )

        return view

    def update_view(
        self,
        observation_id: int,
        panorama: np.ndarray,
    ) -> None:
        """Replace the panorama stored for one observation."""
        self.views[
            observation_id
        ].update(
            panorama
        )

    def reset_stitcher(
        self,
    ) -> None:
        """Restore the preferred panorama stitcher configuration."""
        self._stitcher = Stitcher(
            confidence_threshold=(
                self.best_confidence_threshold
            ),
            detector="sift",
            crop=True,
        )

    def stitch_images(
        self,
        images,
        confidence_threshold: float = 0.9,
    ):
        """Stitch one or more camera images into a panorama."""
        images = list(
            images
        )

        if len(images) == 1:
            image = images[0]

            if (
                image.shape[0] > 1000
                and image.shape[1] > 2000
            ):
                image = cv2.resize(
                    image,
                    (
                        int(
                            image.shape[1]
                            / 4
                        ),
                        int(
                            image.shape[0]
                            / 4
                        ),
                    ),
                )

            return image

        if (
            confidence_threshold
            != self.best_confidence_threshold
        ):
            self._stitcher = Stitcher(
                confidence_threshold=(
                    confidence_threshold
                ),
                detector="sift",
                crop=True,
            )

        try:
            panorama = (
                self._stitcher.stitch(
                    images
                )
            )
        except Exception as error:
            print(
                "[INFO] visual stitching failed:",
                error,
            )
            return None

        return panorama

    @staticmethod
    def compare_images_ssim(
        image_a: np.ndarray,
        image_b: np.ndarray,
    ) -> float:
        """Return rounded structural similarity between two images."""
        resized_b = cv2.resize(
            image_b,
            (
                image_a.shape[1],
                image_a.shape[0],
            ),
            interpolation=cv2.INTER_AREA,
        )

        gray_a = cv2.cvtColor(
            image_a,
            cv2.COLOR_BGR2GRAY,
        )

        gray_b = cv2.cvtColor(
            resized_b,
            cv2.COLOR_BGR2GRAY,
        )

        score, _ = structural_similarity(
            gray_a,
            gray_b,
            full=True,
        )

        return float(
            np.round(
                score,
                2,
            )
        )

    def compare_memory_to_panorama(
        self,
        panorama: np.ndarray,
    ):
        """Compare one panorama against every memorized visual view."""
        scores = []

        for view in self.views:
            score = (
                self.compare_images_ssim(
                    panorama,
                    view.panorama,
                )
            )

            scores.append(
                score
            )

        return scores

    def get_closest_view_id(
        self,
        panorama: np.ndarray,
        matching_threshold=None,
    ):
        """
        Return the closest existing visual observation ID.

        The stored matching threshold remains authoritative to preserve
        baseline lookup behavior.
        """
        scores = (
            self.compare_memory_to_panorama(
                panorama
            )
        )

        if (
            len(scores) == 0
            or max(scores)
            <= self.matching_threshold
        ):
            return None, scores

        return (
            int(
                np.argmax(
                    scores
                )
            ),
            scores,
        )

    def process_images(
        self,
        images,
        confidence_threshold: float = 0.9,
    ):
        """
        Convert camera images into one discrete visual observation.

        A new observation is created when no existing visual view exceeds
        the configured similarity threshold.
        """
        panorama = self.stitch_images(
            images,
            confidence_threshold=(
                confidence_threshold
            ),
        )

        if panorama is None:
            return None, []

        scores = (
            self.compare_memory_to_panorama(
                panorama
            )
        )

        if (
            len(scores) == 0
            or max(scores)
            <= self.matching_threshold
        ):
            self.create_view(
                panorama
            )

            scores.append(
                1.0
            )

        else:
            closest_view_id = int(
                np.argmax(
                    scores
                )
            )

            stored_panorama = (
                self.views[
                    closest_view_id
                ].panorama
            )

            if (
                panorama.shape[1]
                > stored_panorama.shape[1]
            ):
                self.update_view(
                    closest_view_id,
                    panorama,
                )

        observation_id = int(
            np.argmax(
                scores
            )
        )

        return (
            observation_id,
            scores,
        )
