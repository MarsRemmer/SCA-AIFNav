"""Adapt ROS 2 odometry messages to SCA-AIFNav cognitive coordinates."""

from nav_msgs.msg import Odometry

from sca_aifnav_core.baseline_odometry import (
    BaselineOdomTracker,
    CognitiveOdomState,
)
from sca_aifnav_core.planar_geometry import (
    Point2D,
)


class OdometryAdapter:
    """
    Convert physical ROS odometry into the SCA-AIFNav coordinate frame.

    The first valid ROS odometry message establishes the physical origin,
    so the SCA-AIFNav position starts at (0, 0).

    A runtime translation offset can subsequently realign physical
    odometry with a confidently inferred cognitive position. This mirrors
    the odometry-shift semantics used by the reference implementation
    without modifying the original ROS /odom topic.
    """

    def __init__(self) -> None:
        """Create an uninitialized odometry adapter."""
        self._tracker = BaselineOdomTracker()

        self._initialized = False
        self._origin_position = None
        self._latest_raw_position = None

        self._alignment_offset = Point2D(
            0.0,
            0.0,
        )

    @property
    def initialized(self) -> bool:
        """Return whether at least one odometry message was received."""
        return self._initialized

    @property
    def state(self) -> CognitiveOdomState:
        """Return the latest aligned odometry state."""
        return self._tracker.state

    @property
    def origin_position(self):
        """Return the raw ROS position defining the physical origin."""
        return self._origin_position

    @property
    def alignment_offset(self) -> Point2D:
        """Return the current physical-to-cognitive translation."""
        return self._alignment_offset

    def reset(self) -> None:
        """Clear the physical reference and all runtime alignment."""
        self._tracker.reset()

        self._initialized = False
        self._origin_position = None
        self._latest_raw_position = None

        self._alignment_offset = Point2D(
            0.0,
            0.0,
        )

    def update(
        self,
        message: Odometry,
    ) -> CognitiveOdomState:
        """Consume one physical ROS 2 odometry message."""
        if not isinstance(message, Odometry):
            raise TypeError(
                "message must be nav_msgs.msg.Odometry"
            )

        raw_position = Point2D(
            x=float(
                message.pose.pose.position.x
            ),
            y=float(
                message.pose.pose.position.y
            ),
        )

        self._latest_raw_position = raw_position

        if not self._initialized:
            self._initialized = True
            self._origin_position = raw_position

            return self._tracker.reset(
                position=Point2D(
                    0.0,
                    0.0,
                ),
                travel_heading_rad=0.0,
            )

        aligned_position = self._aligned_position(
            raw_position
        )

        return self._tracker.update_position(
            aligned_position
        )

    def realign(
        self,
        cognitive_position: Point2D,
    ) -> CognitiveOdomState:
        """
        Align the current physical odometry with a cognitive position.

        Only planar translation is changed. Physical robot yaw remains
        independent and is handled separately by the navigation node.
        """
        if not isinstance(cognitive_position, Point2D):
            raise TypeError(
                "cognitive_position must be a Point2D"
            )

        if (
            not self._initialized
            or self._origin_position is None
            or self._latest_raw_position is None
        ):
            raise RuntimeError(
                "odometry adapter is not initialized"
            )

        physical_relative_position = (
            self._relative_position(
                self._latest_raw_position
            )
        )

        self._alignment_offset = Point2D(
            x=(
                cognitive_position.x
                - physical_relative_position.x
            ),
            y=(
                cognitive_position.y
                - physical_relative_position.y
            ),
        )

        return self._tracker.reset(
            position=cognitive_position,
            travel_heading_rad=0.0,
        )

    def _relative_position(
        self,
        raw_position: Point2D,
    ) -> Point2D:
        """Convert one raw ROS position to the startup-relative frame."""
        return Point2D(
            x=(
                raw_position.x
                - self._origin_position.x
            ),
            y=(
                raw_position.y
                - self._origin_position.y
            ),
        )

    def _aligned_position(
        self,
        raw_position: Point2D,
    ) -> Point2D:
        """Convert one raw ROS position to the current cognitive frame."""
        relative_position = (
            self._relative_position(
                raw_position
            )
        )

        return Point2D(
            x=(
                relative_position.x
                + self._alignment_offset.x
            ),
            y=(
                relative_position.y
                + self._alignment_offset.y
            ),
        )
