"""Dynamic generative-model state for the SCA-AIFNav baseline."""

import numpy as np

from sca_aifnav_core.probability_tables import (
    create_likelihood_table,
    create_transition_table,
    expand_likelihood_table,
    expand_transition_table,
)


INITIAL_UNKNOWN_LIKELIHOOD = 0.01


class BaselineGenerativeModel:
    """
    Store the dynamic probabilistic model used by the reference baseline.

    The model contains two observation modalities:

    - sensory observations;
    - discrete cognitive-place observations.

    The constructor reproduces the model state after reference baseline
    initialization has associated the starting observations with state 0.
    """

    def __init__(
        self,
        sensory_observations: int = 2,
        place_observations: int = 2,
        num_states: int = 2,
        num_actions: int = 13,
        initial_sensory_observation: int = 0,
        initial_place_observation: int = 0,
        stationary_action_id: int = 12,
    ) -> None:
        """Initialize the baseline-compatible runtime generative model."""
        self.sensory_likelihood = create_likelihood_table(
            sensory_observations,
            num_states,
        )

        self.place_likelihood = create_likelihood_table(
            place_observations,
            num_states,
        )

        self.sensory_concentration = np.ones_like(
            self.sensory_likelihood
        )

        self.place_concentration = np.ones_like(
            self.place_likelihood
        )

        self.transition_likelihood = create_transition_table(
            num_states,
            num_actions,
        )

        self.transition_concentration = (
            self.transition_likelihood.copy()
        )

        self.state_belief = np.full(
            num_states,
            1.0 / num_states,
            dtype=float,
        )

        self.stationary_action_id = stationary_action_id

        self._validate_initial_observation(
            initial_sensory_observation,
            sensory_observations,
            "initial_sensory_observation",
        )
        self._validate_initial_observation(
            initial_place_observation,
            place_observations,
            "initial_place_observation",
        )
        self._validate_stationary_action()

        self._initialize_runtime_likelihoods(
            initial_sensory_observation,
            initial_place_observation,
        )

        self.state_belief = self.infer_state_belief(
            sensory_observation=initial_sensory_observation,
            place_observation=initial_place_observation,
        )

        self._enforce_stationary_transition()

    @property
    def num_states(self) -> int:
        """Return the current hidden-state count."""
        return self.sensory_likelihood.shape[1]

    @property
    def num_actions(self) -> int:
        """Return the number of discrete actions."""
        return self.transition_likelihood.shape[2]

    @property
    def sensory_observations(self) -> int:
        """Return the sensory observation count."""
        return self.sensory_likelihood.shape[0]

    @property
    def place_observations(self) -> int:
        """Return the place observation count."""
        return self.place_likelihood.shape[0]

    def infer_state_belief(
        self,
        sensory_observation=None,
        place_observation=None,
        prior=None,
    ) -> np.ndarray:
        """
        Infer a posterior over the single baseline hidden-state factor.

        Observation likelihoods from the supplied modalities are multiplied
        with the prior and normalized.
        """
        if (
            sensory_observation is None
            and place_observation is None
        ):
            raise ValueError(
                "at least one observation modality is required"
            )

        if prior is None:
            prior = np.full(
                self.num_states,
                1.0 / self.num_states,
                dtype=float,
            )
        else:
            prior = np.asarray(
                prior,
                dtype=float,
            )

            if prior.shape != (self.num_states,):
                raise ValueError(
                    "prior shape must match number of states"
                )

            if not np.all(np.isfinite(prior)):
                raise ValueError(
                    "prior must contain finite values"
                )

        posterior_weight = prior.copy()

        if sensory_observation is not None:
            self._validate_initial_observation(
                sensory_observation,
                self.sensory_observations,
                "sensory_observation",
            )

            posterior_weight *= self.sensory_likelihood[
                sensory_observation,
                :,
            ]

        if place_observation is not None:
            self._validate_initial_observation(
                place_observation,
                self.place_observations,
                "place_observation",
            )

            posterior_weight *= self.place_likelihood[
                place_observation,
                :,
            ]

        total = posterior_weight.sum()

        if total <= 0.0:
            raise ValueError(
                "posterior cannot be normalized"
            )

        return posterior_weight / total

    def predicted_state_prior(
        self,
        action_id: int,
        belief=None,
    ) -> np.ndarray:
        """Predict the next hidden-state prior under one action."""
        self._validate_action_id(action_id)

        if belief is None:
            belief = self.state_belief

        belief = np.asarray(
            belief,
            dtype=float,
        )

        if belief.shape != (self.num_states,):
            raise ValueError(
                "belief shape must match number of states"
            )

        return (
            self.transition_likelihood[
                :,
                :,
                action_id,
            ]
            @ belief
        )

    def register_place_observation(
        self,
        place_id: int,
    ) -> bool:
        """
        Associate a discrete place observation with a hidden state.

        When the place index exceeds the current model size, both hidden
        state and place-observation dimensions expand using baseline rules.

        Return True when the hidden-state dimension grows.
        """
        self._validate_place_id(place_id)

        previous_num_states = self.num_states

        if place_id >= self.num_states:
            dimension_addition = max(
                0,
                place_id + 1 - self.place_observations,
            )

            if dimension_addition > 0:
                self.sensory_likelihood = (
                    expand_likelihood_table(
                        self.sensory_likelihood,
                        add_states=dimension_addition,
                        null_probability=True,
                    )
                )

                self.place_likelihood = (
                    expand_likelihood_table(
                        self.place_likelihood,
                        add_observations=dimension_addition,
                        add_states=dimension_addition,
                        null_probability=True,
                    )
                )

        self.place_likelihood[:, place_id] = 0.0
        self.place_likelihood[
            place_id,
            place_id,
        ] = 1.0

        self._reset_likelihood_concentrations()
        self._synchronize_transition_dimension()

        return self.num_states > previous_num_states

    def _initialize_runtime_likelihoods(
        self,
        sensory_observation: int,
        place_observation: int,
    ) -> None:
        """Reproduce reference baseline likelihood initialization."""
        self.sensory_likelihood[:] = (
            INITIAL_UNKNOWN_LIKELIHOOD
        )
        self.place_likelihood[:] = (
            INITIAL_UNKNOWN_LIKELIHOOD
        )

        self.sensory_likelihood[:, 0] = 0.0
        self.sensory_likelihood[
            sensory_observation,
            0,
        ] = 1.0

        self.place_likelihood[:, 0] = 0.0
        self.place_likelihood[
            place_observation,
            0,
        ] = 1.0

    def _reset_likelihood_concentrations(self) -> None:
        """Reset observation Dirichlet concentrations to one."""
        self.sensory_concentration = np.ones_like(
            self.sensory_likelihood
        )

        self.place_concentration = np.ones_like(
            self.place_likelihood
        )

    def _synchronize_transition_dimension(self) -> None:
        """Expand transition tables and belief when required."""
        transition_states = (
            self.transition_likelihood.shape[0]
        )

        state_addition = (
            self.num_states - transition_states
        )

        if state_addition <= 0:
            return

        self.transition_likelihood = (
            expand_transition_table(
                self.transition_likelihood,
                add_states=state_addition,
                alter_weights=True,
            )
        )

        self.transition_concentration = (
            expand_transition_table(
                self.transition_concentration,
                add_states=state_addition,
                alter_weights=True,
            )
        )

        self.state_belief = np.append(
            self.state_belief,
            np.zeros(state_addition),
        )

    def _enforce_stationary_transition(self) -> None:
        """Make the stationary action an identity transition."""
        self.transition_likelihood[
            :,
            :,
            self.stationary_action_id,
        ] = np.eye(self.num_states)

    def enforce_stationary_transition(self) -> None:
        """Publicly restore the baseline stationary transition."""
        self._enforce_stationary_transition()

    def _validate_stationary_action(self) -> None:
        """Validate the configured stationary action."""
        if (
            isinstance(self.stationary_action_id, bool)
            or not isinstance(
                self.stationary_action_id,
                int,
            )
        ):
            raise TypeError(
                "stationary_action_id must be an integer"
            )

        if not (
            0
            <= self.stationary_action_id
            < self.num_actions
        ):
            raise ValueError(
                "stationary_action_id is out of range"
            )

    def _validate_action_id(
        self,
        action_id: int,
    ) -> None:
        """Validate an action identifier."""
        if isinstance(action_id, bool) or not isinstance(
            action_id,
            int,
        ):
            raise TypeError(
                "action_id must be an integer"
            )

        if not 0 <= action_id < self.num_actions:
            raise ValueError(
                "action_id is out of range"
            )

    @staticmethod
    def _validate_initial_observation(
        observation: int,
        observation_count: int,
        name: str,
    ) -> None:
        """Validate one discrete observation index."""
        if isinstance(observation, bool) or not isinstance(
            observation,
            int,
        ):
            raise TypeError(
                f"{name} must be an integer"
            )

        if not 0 <= observation < observation_count:
            raise ValueError(
                f"{name} is out of range"
            )

    @staticmethod
    def _validate_place_id(place_id: int) -> None:
        """Validate a discrete place identifier."""
        if isinstance(place_id, bool) or not isinstance(
            place_id,
            int,
        ):
            raise TypeError(
                "place_id must be an integer"
            )

        if place_id < 0:
            raise ValueError(
                "place_id must be non-negative"
            )
