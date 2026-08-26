"""Adapt ROS 2 odometry messages to cognitive odometry."""

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
    Convert ROS odometry into origin-relative cognitive odometry.

    The first valid ROS odometry position establishes the physical
    reference origin. Cognitive XY therefore starts at (0, 0), while
    subsequent states contain displacement relative to that first pose.

    Cognitive heading represents displacement direction rather than the
    physical robot body orientation.
    """

    def __init__(self) -> None:
        """Create an uninitialized odometry adapter."""
        self._tracker = BaselineOdomTracker()
        self._initialized = False
        self._origin_position = None

    @property
    def initialized(self) -> bool:
        """Return whether at least one odometry message was received."""
        return self._initialized

    @property
    def state(self) -> CognitiveOdomState:
        """Return the latest cognitive odometry state."""
        return self._tracker.state

    @property
    def origin_position(self):
        """Return the raw ROS position defining the cognitive origin."""
        return self._origin_position

    def reset(self) -> None:
        """Clear the ROS reference and restore cognitive origin state."""
        self._tracker.reset()
        self._initialized = False
        self._origin_position = None

    def update(
        self,
        message: Odometry,
    ) -> CognitiveOdomState:
        """Consume one ROS odometry message."""
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

        relative_position = Point2D(
            x=(
                raw_position.x
                - self._origin_position.x
            ),
            y=(
                raw_position.y
                - self._origin_position.y
            ),
        )

        return self._tracker.update_position(
            relative_position
        )
