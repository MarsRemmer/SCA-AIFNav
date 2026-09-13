"""Bridge completed ROS observations into the navigation core."""

from dataclasses import dataclass
from typing import Optional

from sca_aifnav_core.generative_model import (
    BaselineGenerativeModel,
)
from sca_aifnav_core.motion_primitives import (
    BaselineMotionSet,
)
from sca_aifnav_core.navigation_mode import (
    EXPLORE,
    GOAL_BALANCED,
    GOAL_DIRECT,
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
    goal_reached: bool = False


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
                    max_lookahead_steps=8,
                    max_rollout_depth=10,
                )
            )

            # AIMAPP exploration baseline:
            # state information gain on, utility and inductive
            # inference off.
            coordinator.set_navigation_mode(
                EXPLORE
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
        self._goal_reached = False

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

    @property
    def navigation_mode(
        self,
    ):
        """Return the currently selected navigation mode."""
        return getattr(
            self.coordinator,
            "navigation_mode",
            None,
        )

    @property
    def goal_reached(
        self,
    ) -> bool:
        """Return whether the current goal task has completed."""
        return self._goal_reached

    def _ensure_mode_change_is_safe(
        self,
    ) -> None:
        """Reject a mode change while action learning is incomplete."""
        if self._completed_action_id is not None:
            raise RuntimeError(
                "cannot change navigation mode while "
                "a completed physical action is "
                "awaiting observation"
            )

    def _observation_matches_goal(
        self,
        observation: NavigationObservation,
    ) -> bool:
        """Return whether one observation satisfies the current goal."""
        if self.navigation_mode not in (
            GOAL_DIRECT,
            GOAL_BALANCED,
        ):
            return False

        (
            preferred_sensory,
            preferred_place,
        ) = (
            self.coordinator
            .preferences
            .preferred_observations
        )

        matches = []

        if preferred_sensory >= 0:
            matches.append(
                observation.sensory_observation
                == preferred_sensory
            )

        if preferred_place >= 0:
            matches.append(
                observation.place_observation
                == preferred_place
            )

        return bool(matches) and all(
            matches
        )

    def _replan_after_mode_change(
        self,
    ):
        """Replace the unexecuted plan after changing navigation mode."""
        if self._latest_decision is None:
            self._planned_action_id = None
            self._failure_possible_actions = None
            return None

        source_place_id = (
            self._planning_source_place_id()
        )

        observation = (
            self._latest_decision.observation
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
                    source_place_id
                ),
                obstacle_distances=(
                    observation.obstacle_distances
                ),
            )

        planning = self.coordinator.plan_current(
            current_place_id=(
                source_place_id
            ),
            possible_actions=possible_actions,
        )

        goal_reached = (
            self._observation_matches_goal(
                observation
            )
        )

        if goal_reached:
            selected_action = None
        else:
            selected_action = int(
                planning.selected_action
            )

        previous_decision = (
            self._latest_decision
        )

        previous_cycle = (
            previous_decision.cycle_result
        )

        cycle_result = NavigationCycleResult(
            learning=(
                previous_cycle.learning
            ),
            preferences=(
                self.coordinator
                .preferences
                .snapshot()
            ),
            planning=planning,
            posterior_place_id=getattr(
                previous_cycle,
                "posterior_place_id",
                -1,
            ),
        )

        decision = NavigationCoreDecision(
            observation=(
                previous_decision.observation
            ),
            # A mode change performs planning only. It must not
            # expose the previously learned physical action as newly
            # executed evidence.
            executed_action_id=None,
            next_action_id=selected_action,
            cycle_result=cycle_result,
            is_bootstrap=(
                previous_decision.is_bootstrap
            ),
            goal_reached=goal_reached,
        )

        self._planned_action_id = (
            selected_action
        )

        self._goal_reached = goal_reached
        self._failure_possible_actions = None
        self._latest_decision = decision

        return decision

    def set_exploration_navigation(
        self,
    ):
        """Enter EXPLORE mode and immediately replan when initialized."""
        self._ensure_mode_change_is_safe()

        self.coordinator.set_exploration_navigation()
        self._goal_reached = False

        return self._replan_after_mode_change()

    def set_goal_navigation(
        self,
        mode: str,
        sensory_observation: int = -1,
        place_observation: int = -1,
        preference_weight: float = 10.0,
    ):
        """Set a known goal, select a goal mode, and replan."""
        if mode not in (
            GOAL_DIRECT,
            GOAL_BALANCED,
        ):
            raise ValueError(
                "goal navigation mode must be "
                "goal_direct or goal_balanced"
            )

        self._ensure_mode_change_is_safe()

        self.coordinator.set_goal_navigation(
            mode=mode,
            sensory_observation=(
                sensory_observation
            ),
            place_observation=(
                place_observation
            ),
            preference_weight=(
                preference_weight
            ),
        )

        self._goal_reached = False

        return self._replan_after_mode_change()

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

        goal_reached = (
            self._observation_matches_goal(
                observation
            )
        )

        if goal_reached:
            next_action_id = None
        else:
            next_action_id = (
                cycle_result.planning.selected_action
            )

        self._completed_action_id = None
        self._planned_action_id = (
            next_action_id
        )

        self._goal_reached = goal_reached
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
            goal_reached=goal_reached,
        )

        self._latest_decision = decision

        return decision

    def _planning_source_place_id(
        self,
    ) -> int:
        """Return the cognitive place from which the current plan starts."""
        if self._latest_decision is None:
            raise RuntimeError(
                "no navigation decision exists"
            )

        cycle_result = (
            self._latest_decision
            .cycle_result
        )

        posterior_place_id = getattr(
            cycle_result,
            "posterior_place_id",
            -1,
        )

        if posterior_place_id >= 0:
            return int(
                posterior_place_id
            )

        planning = getattr(
            cycle_result,
            "planning",
            None,
        )

        root_node = getattr(
            planning,
            "root_node",
            None,
        )

        if root_node is not None:
            return int(
                root_node.place_id
            )

        # Compatibility fallback for lightweight test/fake planning
        # objects that do not expose a real MCTS root.
        return int(
            self._latest_decision
            .observation
            .place_observation
        )

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
            self._planning_source_place_id()
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

        if not any(
            self.motion_set.is_directional(candidate)
            for candidate in self._failure_possible_actions
        ):
            raise RuntimeError(
                "no directional navigation actions remain "
                "after failures"
            )

        source_place_id = (
            self._planning_source_place_id()
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
