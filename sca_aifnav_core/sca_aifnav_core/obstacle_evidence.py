"""Obstacle-driven transition evidence for the baseline."""

import numpy as np

from sca_aifnav_core.generative_model import BaselineGenerativeModel
from sca_aifnav_core.imagined_linking import (
    IMAGINED_DIRECT_RATE,
    IMAGINED_REVERSE_RATE,
)
from sca_aifnav_core.motion_primitives import BaselineMotionSet
from sca_aifnav_core.transition_learning import (
    DIRECT_TRANSITION_RATE,
    REVERSE_TRANSITION_RATE,
    learn_bidirectional_transition,
    learn_transition,
)


BLOCKED_SELF_LOOP_RATE = 10.0
UNREACHABLE_DIRECT_RATE = -DIRECT_TRANSITION_RATE
UNREACHABLE_REVERSE_RATE = -REVERSE_TRANSITION_RATE


def reinforce_blocked_self_loop(
    model: BaselineGenerativeModel,
    belief: np.ndarray,
    action_id: int,
    motion_set: BaselineMotionSet,
) -> np.ndarray:
    """
    Reinforce staying in the current state when an action is blocked.

    The baseline uses a +10 transition update for the direct first-step
    obstacle case.
    """
    _validate_directional_action(
        action_id,
        motion_set,
    )

    return learn_transition(
        model=model,
        current_belief=belief,
        previous_belief=belief,
        action_id=action_id,
        learning_rate=BLOCKED_SELF_LOOP_RATE,
    )


def discourage_known_unreachable_link(
    model: BaselineGenerativeModel,
    current_belief: np.ndarray,
    unreachable_belief: np.ndarray,
    action_id: int,
    motion_set: BaselineMotionSet,
) -> int:
    """
    Reduce belief in a known transition that proved unreachable.

    baseline applies -10 in the direct direction and -7 in the reverse
    direction.
    """
    _validate_directional_action(
        action_id,
        motion_set,
    )

    return learn_bidirectional_transition(
        model=model,
        previous_belief=current_belief,
        next_belief=unreachable_belief,
        action_id=action_id,
        motion_set=motion_set,
        direct_rate=UNREACHABLE_DIRECT_RATE,
        reverse_rate=UNREACHABLE_REVERSE_RATE,
    )


def apply_direct_imagined_evidence(
    model: BaselineGenerativeModel,
    previous_belief: np.ndarray,
    next_belief: np.ndarray,
    action_id: int,
    motion_set: BaselineMotionSet,
    reachable: bool,
) -> int:
    """
    Reinforce or weaken one direct imagined transition.

    A reachable imagined edge uses +5/+3. An imagined edge contradicted
    by obstacle information uses -5/-3.
    """
    _validate_directional_action(
        action_id,
        motion_set,
    )

    sign = 1.0 if reachable else -1.0

    return learn_bidirectional_transition(
        model=model,
        previous_belief=previous_belief,
        next_belief=next_belief,
        action_id=action_id,
        motion_set=motion_set,
        direct_rate=sign * IMAGINED_DIRECT_RATE,
        reverse_rate=sign * IMAGINED_REVERSE_RATE,
    )


def _validate_directional_action(
    action_id: int,
    motion_set: BaselineMotionSet,
) -> None:
    """Require one non-stationary directional action."""
    motion_set.action(action_id)

    if not motion_set.is_directional(action_id):
        raise ValueError(
            "obstacle evidence requires a directional action"
        )
