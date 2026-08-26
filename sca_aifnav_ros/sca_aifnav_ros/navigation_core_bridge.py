"""Bridge completed ROS observations into the navigation core."""

from dataclasses import dataclass
from typing import Optional

from sca_aifnav_core.generative_model import (
    BaselineGenerativeModel,
)
from sca_aifnav_core.motion_primitives import (
    BaselineMotionSet,
)
from sca_aifnav_core.navigation_cycle import (
    BaselineNavigationCoordinator,
    NavigationCycleResult,
)
from sca_aifnav_core.planar_geometry import (
    Point2D,
)
from sca_aifnav_core.spatial_memory import (
    BaselinePlaceMemory,
)
from sca_aifnav_ros.navigation_observation import (
    NavigationObservation,
)


BOOTSTRAP_ACTION_ID = (
    BaselineMotionSet.ACTION_COUNT - 1
)


@dataclass(frozen=True)
class NavigationCoreDecision:
    """Summarize one observation-to-planning core cycle."""

    observation: NavigationObservation
    executed_action_id: int
    next_action_id: Optional[int]
    cycle_result: NavigationCycleResult
    is_bootstrap: bool


@dataclass(frozen=True)
class NavigationActionTarget:
    """Describe the cognitive target of one planned physical action."""

    action_id: int
    source_place_id: int
    target_place_id: int
    target_position: Point2D
    is_stationary: bool


class NavigationCoreBridge:
    """Manage observation, executed-action, and planning lifecycle."""

    def __init__(
        self,
        memory=None,
        model=None,
        motion_set=None,
        coordinator=None,
    ) -> None:
        """Create a persistent navigation-core bridge."""
        if memory is None:
            memory = BaselinePlaceMemory()

        if model is None:
            model = BaselineGenerativeModel()

        if motion_set is None:
            motion_set = BaselineMotionSet()

        if coordinator is None:
            coordinator = (
                BaselineNavigationCoordinator(
                    model=model,
                    memory=memory,
                    motion_set=motion_set,
                )
            )

        self.memory = memory
        self.model = model
        self.motion_set = motion_set
        self.coordinator = coordinator

        self._cycle_count = 0
        self._planned_action_id = None
        self._completed_action_id = None
        self._latest_decision = None

    @property
    def is_initialized(
        self,
    ) -> bool:
        """Return whether at least one core cycle has completed."""
        return self._cycle_count > 0

    @property
    def next_action_id(
        self,
    ):
        """Return the currently planned but uncompleted action."""
        return self._planned_action_id

    @property
    def completed_action_id(
        self,
    ):
        """Return the action awaiting learning at the next observation."""
        return self._completed_action_id

    @property
    def latest_decision(
        self,
    ):
        """Return the most recently completed core decision."""
        return self._latest_decision

    def process_observation(
        self,
        observation: NavigationObservation,
    ) -> NavigationCoreDecision:
        """Learn from one observation and plan the following action."""
        if not isinstance(
            observation,
            NavigationObservation,
        ):
            raise TypeError(
                "observation must be a NavigationObservation"
            )

        is_bootstrap = (
            self._cycle_count == 0
        )

        if is_bootstrap:
            executed_action_id = (
                BOOTSTRAP_ACTION_ID
            )
        else:
            if self._completed_action_id is None:
                raise RuntimeError(
                    "no completed physical action is "
                    "available for this observation"
                )

            executed_action_id = (
                self._completed_action_id
            )

        cycle_result = (
            self.coordinator.step_and_plan(
                state=observation.state,
                sensory_observation=(
                    observation.sensory_observation
                ),
                place_observation=(
                    observation.place_observation
                ),
                executed_action_id=(
                    executed_action_id
                ),
                obstacle_distances=(
                    observation.obstacle_distances
                ),
                current_place_id=(
                    observation.place_observation
                ),
            )
        )

        next_action_id = (
            cycle_result.planning.selected_action
        )

        self._completed_action_id = None
        self._planned_action_id = (
            next_action_id
        )
        self._cycle_count += 1

        decision = NavigationCoreDecision(
            observation=observation,
            executed_action_id=(
                executed_action_id
            ),
            next_action_id=(
                next_action_id
            ),
            cycle_result=(
                cycle_result
            ),
            is_bootstrap=is_bootstrap,
        )

        self._latest_decision = decision

        return decision

    def resolve_planned_action_target(
        self,
    ):
        """Resolve the currently planned action to its cognitive target."""
        if (
            self._latest_decision is None
            or self._planned_action_id is None
        ):
            return None

        action_id = self._planned_action_id

        source_place_id = (
            self._latest_decision
            .observation
            .place_observation
        )

        target_place_id = (
            self.coordinator
            .model_interface
            .get_next_place_id(
                current_place_id=(
                    source_place_id
                ),
                action_id=action_id,
            )
        )

        if target_place_id < 0:
            raise RuntimeError(
                "planned action does not resolve "
                "to a known cognitive place"
            )

        target_position = self.memory.place(
            target_place_id
        )

        if target_position is None:
            raise RuntimeError(
                "planned target place is missing "
                "from cognitive memory"
            )

        primitive = self.motion_set.action(
            action_id
        )

        return NavigationActionTarget(
            action_id=action_id,
            source_place_id=(
                source_place_id
            ),
            target_place_id=(
                target_place_id
            ),
            target_position=(
                target_position
            ),
            is_stationary=(
                primitive.is_stationary
            ),
        )

    def record_executed_action(
        self,
        action_id: int,
    ) -> None:
        """Record completion of the currently planned physical action."""
        if (
            isinstance(action_id, bool)
            or not isinstance(action_id, int)
        ):
            raise TypeError(
                "action_id must be an integer"
            )

        if self._planned_action_id is None:
            raise RuntimeError(
                "no planned action is awaiting execution"
            )

        if action_id != self._planned_action_id:
            raise ValueError(
                "executed action does not match "
                "the planned action"
            )

        self._completed_action_id = (
            action_id
        )
        self._planned_action_id = None
