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


@dataclass(frozen=True)
class NavigationCoreDecision:
    """Summarize one observation-to-planning core cycle."""

    observation: NavigationObservation
    executed_action_id: int | None
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
                    robot_dimension=0.3,
                    use_utility=False,
                    use_state_information_gain=True,
                    use_inductive_inference=False,
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
        self._failure_possible_actions = None

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

    def _initialize_and_plan(
        self,
        observation: NavigationObservation,
    ) -> NavigationCycleResult:
        """Initialize the model before any physical action has occurred."""
        from sca_aifnav_core.transition_node_update import (
            update_cognitive_transition_nodes,
        )

        update_cognitive_transition_nodes(
            state=observation.state,
            obstacle_distances=(
                observation.obstacle_distances
            ),
            memory=self.memory,
            model=self.model,
            motion_set=self.motion_set,
            robot_dimension=(
                self.coordinator.learning.robot_dimension
            ),
            max_steps=(
                self.coordinator.learning
                .max_lookahead_steps
            ),
        )

        self.coordinator.learning.history.align_to_states(
            self.model.num_states
        )

        self.model.enforce_stationary_transition()

        self.coordinator.preferences.sync_dimensions(
            self.model
        )

        preferences = (
            self.coordinator.preferences.snapshot()
        )

        restrictive_actions = getattr(
            self.coordinator,
            "restrictive_possible_actions",
            None,
        )

        possible_actions = None

        if callable(restrictive_actions):
            possible_actions = restrictive_actions(
                current_place_id=(
                    observation.place_observation
                ),
                obstacle_distances=(
                    observation.obstacle_distances
                ),
            )

        planning = self.coordinator.plan_current(
            current_place_id=(
                observation.place_observation
            ),
            possible_actions=possible_actions,
        )

        return NavigationCycleResult(
            learning=None,
            preferences=preferences,
            planning=planning,
        )

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
            executed_action_id = None

            cycle_result = (
                self._initialize_and_plan(
                    observation
                )
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

        self._failure_possible_actions = None

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

    @property
    def remaining_retry_actions(
        self,
    ):
        """Return actions still available after physical failures."""
        if self._failure_possible_actions is None:
            return None

        return tuple(
            self._failure_possible_actions
        )

    def record_failed_action(
        self,
        action_id: int,
    ) -> NavigationActionTarget:
        """Learn negative transition evidence for a failed action."""
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
                "failed action does not match "
                "the planned action"
            )

        if not self.motion_set.is_directional(
            action_id
        ):
            raise ValueError(
                "only directional navigation actions can fail"
            )

        target = (
            self.resolve_planned_action_target()
        )

        if target is None:
            raise RuntimeError(
                "failed action has no cognitive target"
            )

        from sca_aifnav_core.obstacle_evidence import (
            discourage_known_unreachable_link,
        )

        current_belief = (
            self.model.state_belief.copy()
        )

        predicted_prior = (
            self.model.predicted_state_prior(
                action_id=action_id,
                belief=current_belief,
            )
        )

        unreachable_belief = (
            self.model.infer_state_belief(
                place_observation=(
                    target.target_place_id
                ),
                prior=predicted_prior,
            )
        )

        discourage_known_unreachable_link(
            model=self.model,
            current_belief=current_belief,
            unreachable_belief=(
                unreachable_belief
            ),
            action_id=action_id,
            motion_set=self.motion_set,
        )

        if self._failure_possible_actions is None:
            planning = (
                self._latest_decision
                .cycle_result
                .planning
            )

            available_actions = getattr(
                planning,
                "available_actions",
                None,
            )

            if available_actions is None:
                raise RuntimeError(
                    "current planning result does not expose "
                    "available actions"
                )

            self._failure_possible_actions = [
                int(candidate)
                for candidate in available_actions
            ]

        self._failure_possible_actions = [
            candidate
            for candidate
            in self._failure_possible_actions
            if candidate != action_id
        ]

        self._planned_action_id = None

        return target

    def replan_after_failed_action(
        self,
    ):
        """Replan from the previous place using untried actions only."""
        if self._latest_decision is None:
            raise RuntimeError(
                "no navigation decision exists for replanning"
            )

        if self._planned_action_id is not None:
            raise RuntimeError(
                "the failed action must be cleared "
                "before replanning"
            )

        if self._failure_possible_actions is None:
            raise RuntimeError(
                "no failed-action retry lifecycle is active"
            )

        if not self._failure_possible_actions:
            raise RuntimeError(
                "no navigation actions remain after failures"
            )

        source_place_id = (
            self._latest_decision
            .observation
            .place_observation
        )

        planning = self.coordinator.plan_current(
            current_place_id=(
                source_place_id
            ),
            possible_actions=tuple(
                self._failure_possible_actions
            ),
        )

        selected_action = int(
            planning.selected_action
        )

        if (
            selected_action
            not in self._failure_possible_actions
        ):
            raise RuntimeError(
                "replanned action is outside "
                "the remaining action set"
            )

        self._planned_action_id = (
            selected_action
        )

        return planning

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
