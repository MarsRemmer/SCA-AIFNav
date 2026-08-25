"""Model interface for the active inference MCTS planner."""

from dataclasses import replace

from sca_aifnav_core.action_evaluation import (
    ActionEvaluation,
    ExpectedObservations,
    evaluate_action,
    predict_action_state,
    predict_observations,
)
from sca_aifnav_core.baseline_odometry import (
    CognitiveOdomState,
)
from sca_aifnav_core.cognitive_projection import (
    position_in_action_sector,
    project_action_position,
)
from sca_aifnav_core.generative_model import (
    BaselineGenerativeModel,
)
from sca_aifnav_core.inductive_inference import (
    DEFAULT_INDUCTIVE_HORIZON,
    inductive_bonus,
)
from sca_aifnav_core.motion_primitives import (
    BaselineMotionSet,
)
from sca_aifnav_core.preference_state import (
    BaselinePreferenceState,
)
from sca_aifnav_core.spatial_memory import (
    BaselinePlaceMemory,
)


class MCTSModelInterface:
    """Expose the baseline generative model to Monte Carlo tree search."""

    def __init__(
        self,
        model: BaselineGenerativeModel,
        memory: BaselinePlaceMemory,
        motion_set: BaselineMotionSet,
        preferences: BaselinePreferenceState,
        use_utility: bool = True,
        use_state_information_gain: bool = True,
        use_inductive_inference: bool = False,
        inductive_horizon: int = DEFAULT_INDUCTIVE_HORIZON,
    ) -> None:
        self.model = model
        self.memory = memory
        self.motion_set = motion_set
        self.preferences = preferences

        self.use_utility = bool(
            use_utility
        )

        self.use_state_information_gain = bool(
            use_state_information_gain
        )

        self.use_inductive_inference = bool(
            use_inductive_inference
        )

        self.inductive_horizon = inductive_horizon

    def get_possible_actions(self):
        """Return every baseline action, including STAY."""
        return list(
            range(self.model.num_actions)
        )

    def get_place_position(
        self,
        place_id: int,
    ):
        """Return the stored position of one cognitive place."""
        self._validate_place_id(
            place_id
        )

        return self.memory.place(
            place_id
        )

    def get_next_place_id(
        self,
        current_place_id: int,
        action_id: int,
    ) -> int:
        """
        Return the known place reached by one baseline action.

        MCTS never creates cognitive places. It first projects the action,
        searches the existing fixed-radius place memory, and finally checks
        that the resolved place really lies inside that action's range.
        """
        self._validate_place_id(
            current_place_id
        )

        action = self.motion_set.action(
            action_id
        )

        current_position = self.memory.place(
            current_place_id
        )

        if action.is_stationary:
            return current_place_id

        reference_state = CognitiveOdomState(
            position=current_position,
            travel_heading_rad=0.0,
        )

        projected = project_action_position(
            state=reference_state,
            action_id=action_id,
            memory=self.memory,
            motion_set=self.motion_set,
        )

        next_place_id = self.memory.find_match(
            projected
        )

        if next_place_id < 0:
            return -1

        next_position = self.memory.place(
            next_place_id
        )

        in_action_range = position_in_action_sector(
            origin=current_position,
            candidate=next_position,
            action_id=action_id,
            influence_radius=(
                self.memory.influence_radius
            ),
            motion_set=self.motion_set,
        )

        if not in_action_range:
            return -1

        return next_place_id

    def get_next_state_belief(
        self,
        current_belief,
        action_id: int,
    ):
        """Predict the next state belief using B."""
        return predict_action_state(
            model=self.model,
            state_belief=current_belief,
            action_id=action_id,
        )

    def get_expected_observation(
        self,
        state_belief,
    ) -> ExpectedObservations:
        """Predict expected sensory and place observations using A."""
        return predict_observations(
            model=self.model,
            state_belief=state_belief,
        )

    def evaluate_action(
        self,
        current_belief,
        action_id: int,
    ) -> ActionEvaluation:
        """Calculate all enabled baseline reward terms for one action."""
        base_evaluation = evaluate_action(
            model=self.model,
            preferences=self.preferences,
            action_id=action_id,
            state_belief=current_belief,
            use_utility=self.use_utility,
            use_state_information_gain=(
                self.use_state_information_gain
            ),
        )

        if not self.use_inductive_inference:
            return base_evaluation

        bonus = inductive_bonus(
            current_belief=current_belief,
            predicted_belief=(
                base_evaluation.predicted_state
            ),
            transition_likelihood=(
                self.model.transition_likelihood
            ),
            preferred_states=(
                self.preferences.preferred_states
            ),
            lookahead_horizon=(
                self.inductive_horizon
            ),
        )

        return replace(
            base_evaluation,
            score=(
                base_evaluation.score
                + bonus
            ),
        )

    def _validate_place_id(
        self,
        place_id: int,
    ) -> None:
        """Validate one cognitive-place identifier."""
        if (
            isinstance(place_id, bool)
            or not isinstance(place_id, int)
        ):
            raise TypeError(
                "place_id must be an integer"
            )

        if not 0 <= place_id < len(self.memory):
            raise ValueError(
                "place_id is out of range"
            )
