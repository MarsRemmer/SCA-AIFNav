"""ROS 2 image adaptation for SCA-AIFNav visual perception."""

import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import Image


class ImageAdapter:
    """Convert ROS 2 image messages into OpenCV BGR images."""

    def __init__(self) -> None:
        """Create the ROS image bridge."""
        self._bridge = CvBridge()

    def to_bgr(
        self,
        message: Image,
    ) -> np.ndarray:
        """Convert one ROS Image message to an OpenCV BGR image."""
        if not isinstance(
            message,
            Image,
        ):
            raise TypeError(
                "message must be sensor_msgs.msg.Image"
            )

        image = self._bridge.imgmsg_to_cv2(
            message,
            desired_encoding="bgr8",
        )

        if not isinstance(
            image,
            np.ndarray,
        ):
            raise TypeError(
                "converted image must be a numpy array"
            )

        if image.ndim != 3:
            raise ValueError(
                "converted image must have three dimensions"
            )

        if image.shape[2] != 3:
            raise ValueError(
                "converted image must have three BGR channels"
            )

        return image
