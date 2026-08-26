"""Tests for failed-action model correction and replanning."""

from types import SimpleNamespace

import numpy as np

from sca_aifnav_core.baseline_odometry import (
    CognitiveOdomState,
)
from sca_aifnav_core.generative_model import (
    BaselineGenerativeModel,
)
from sca_aifnav_core.motion_primitives import (
    BaselineMotionSet,
)
from sca_aifnav_core.planar_geometry import (
    Point2D,
)
from sca_aifnav_core.spatial_memory import (
    BaselinePlaceMemory,
)
from sca_aifnav_ros.navigation_core_bridge import (
    NavigationCoreBridge,
)
from sca_aifnav_ros.navigation_observation import (
    NavigationObservation,
)


class FakeHistory:
    """Provide bootstrap history alignment."""

    def align_to_states(
        self,
        num_states,
    ):
        """Accept state dimensionality."""
        return None


class FakeLearning:
    """Provide bootstrap geometry settings."""

    robot_dimension = 0.25
    max_lookahead_steps = 8

    def __init__(
        self,
    ):
        self.history = FakeHistory()


class FakePreferences:
    """Provide bootstrap preference interfaces."""

    def sync_dimensions(
        self,
        model,
    ):
        """Accept dimensional synchronization."""
        return None

    def snapshot(
        self,
    ):
        """Return a minimal preference snapshot."""
        return None


class FakeModelInterface:
    """Resolve deterministic cognitive targets."""

    def __init__(
        self,
        source_id,
        target_id,
    ):
        self.source_id = source_id
        self.target_id = target_id

    def get_next_place_id(
        self,
        current_place_id,
        action_id,
    ):
        """Resolve actions used by this test."""
        if action_id == 0:
            return self.target_id

        return self.source_id


class FakeCoordinator:
    """Provide deterministic failure-retry planning."""

    def __init__(
        self,
        source_id,
        target_id,
    ):
        self.learning = FakeLearning()
        self.preferences = FakePreferences()

        self.model_interface = (
            FakeModelInterface(
                source_id=source_id,
                target_id=target_id,
            )
        )

        self.calls = []

    def plan_current(
        self,
        current_place_id,
        possible_actions=None,
        **kwargs,
    ):
        """Select action zero initially, then the first remaining action."""
        self.calls.append(
            possible_actions
        )

        if possible_actions is None:
            available = (
                0,
                1,
            )
        else:
            available = tuple(
                possible_actions
            )

        return SimpleNamespace(
            selected_action=available[0],
            available_actions=available,
        )


def observation():
    """Create the source observation for one navigation attempt."""
    return NavigationObservation(
        state=CognitiveOdomState(
            position=Point2D(
                0.0,
                0.0,
            ),
            travel_heading_rad=0.0,
        ),
        sensory_observation=0,
        place_observation=0,
        obstacle_distances=tuple(
            3.5
            for _ in range(12)
        ),
        odometry_revision=1,
        scan_revision=1,
    )


def configured_bridge():
    """Create a bridge with one known directional target."""
    memory = BaselinePlaceMemory()

    source_id = memory.resolve_place(
        Point2D(
            0.0,
            0.0,
        )
    )

    target_id = memory.resolve_place(
        Point2D(
            0.6,
            0.16,
        )
    )

    model = BaselineGenerativeModel()

    model.register_place_observation(
        target_id
    )

    model.state_belief = np.array(
        [
            1.0,
            0.0,
        ]
    )

    # Give every transition positive support and sufficiently
    # large concentration so negative evidence is measurable.
    model.transition_concentration[:] = 20.0

    denominator = (
        model.transition_concentration.sum(
            axis=0,
            keepdims=True,
        )
    )

    model.transition_likelihood = (
        model.transition_concentration
        / denominator
    )

    motion_set = BaselineMotionSet()

    coordinator = FakeCoordinator(
        source_id=source_id,
        target_id=target_id,
    )

    bridge = NavigationCoreBridge(
        model=model,
        memory=memory,
        motion_set=motion_set,
        coordinator=coordinator,
    )

    return (
        bridge,
        coordinator,
        source_id,
        target_id,
    )


def test_failed_action_reduces_transition_evidence():
    """A failed known edge should receive direct and reverse negative evidence."""
    (
        bridge,
        coordinator,
        source_id,
        target_id,
    ) = configured_bridge()

    bridge.process_observation(
        observation()
    )

    direct_before = (
        bridge.model.transition_concentration[
            target_id,
            source_id,
            0,
        ]
    )

    reverse_action = (
        bridge.motion_set.reverse_action(
            0
        )
    )

    reverse_before = (
        bridge.model.transition_concentration[
            source_id,
            target_id,
            reverse_action,
        ]
    )

    failed_target = bridge.record_failed_action(
        0
    )

    assert failed_target.source_place_id == source_id
    assert failed_target.target_place_id == target_id

    assert (
        bridge.model.transition_concentration[
            target_id,
            source_id,
            0,
        ]
        < direct_before
    )

    assert (
        bridge.model.transition_concentration[
            source_id,
            target_id,
            reverse_action,
        ]
        < reverse_before
    )

    assert bridge.next_action_id is None
    assert bridge.remaining_retry_actions == (1,)


def test_replanning_excludes_failed_action():
    """Retry planning should receive only actions not yet failed."""
    (
        bridge,
        coordinator,
        source_id,
        target_id,
    ) = configured_bridge()

    bridge.process_observation(
        observation()
    )

    bridge.record_failed_action(
        0
    )

    planning = (
        bridge.replan_after_failed_action()
    )

    assert planning.selected_action == 1
    assert bridge.next_action_id == 1

    assert (
        coordinator.calls[-1]
        == (
            1,
        )
    )
