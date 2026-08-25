"""Adapt ROS 2 odometry messages to SCA-AIFNav cognitive odometry."""

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
    Convert ROS 2 odometry positions into cognitive odometry states.

    The first valid message establishes the position reference without
    inventing a displacement. Later messages use consecutive planar
    positions to infer the direction of actual travel.

    Robot body orientation from the ROS odometry quaternion is
    intentionally ignored because the cognitive heading represents
    displacement direction rather than physical yaw.
    """

    def __init__(self) -> None:
        """Create an uninitialized odometry adapter."""
        self._tracker = BaselineOdomTracker()
        self._initialized = False

    @property
    def initialized(self) -> bool:
        """Return whether at least one odometry message was received."""
        return self._initialized

    @property
    def state(self) -> CognitiveOdomState:
        """Return the latest cognitive odometry state."""
        return self._tracker.state

    def reset(self) -> None:
        """Clear the ROS odometry reference and restore the origin state."""
        self._tracker.reset()
        self._initialized = False

    def update(
        self,
        message: Odometry,
    ) -> CognitiveOdomState:
        """Consume one ROS 2 odometry message."""
        if not isinstance(message, Odometry):
            raise TypeError(
                "message must be nav_msgs.msg.Odometry"
            )

        position = Point2D(
            x=float(
                message.pose.pose.position.x
            ),
            y=float(
                message.pose.pose.position.y
            ),
        )

        if not self._initialized:
            self._initialized = True

            return self._tracker.reset(
                position=position,
                travel_heading_rad=0.0,
            )

        return self._tracker.update_position(
            position
        )
