"""Tests for posterior cognitive-pose correction."""

from types import SimpleNamespace

import pytest

from sca_aifnav_core.baseline_odometry import (
    BaselineOdomTracker,
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
from sca_aifnav_ros.navigation_node import (
    NavigationNode,
)


class RecordingModelInterface:
    """Record the place used to resolve one planned action."""

    def __init__(
        self,
        target_place_id,
    ):
        self.target_place_id = target_place_id
        self.calls = []

    def get_next_place_id(
        self,
        current_place_id,
        action_id,
    ):
        self.calls.append(
            (
                current_place_id,
                action_id,
            )
        )

        return self.target_place_id


class RecordingCoordinator:
    """Minimal coordinator used by bridge source-place tests."""

    def __init__(
        self,
        model_interface,
    ):
        self.model_interface = model_interface
        self.plan_calls = []

    def plan_current(
        self,
        current_place_id,
        possible_actions=None,
    ):
        self.plan_calls.append(
            (
                current_place_id,
                possible_actions,
            )
        )

        return SimpleNamespace(
            selected_action=1,
        )


def make_bridge_with_corrected_root():
    """Create a bridge whose latest MCTS root differs from observation."""
    memory = BaselinePlaceMemory()

    memory.resolve_place(
        Point2D(
            0.0,
            0.0,
        )
    )

    memory.resolve_place(
        Point2D(
            1.0,
            0.0,
        )
    )

    memory.resolve_place(
        Point2D(
            2.0,
            0.0,
        )
    )

    interface = RecordingModelInterface(
        target_place_id=2
    )

    coordinator = RecordingCoordinator(
        model_interface=interface
    )

    bridge = NavigationCoreBridge.__new__(
        NavigationCoreBridge
    )

    bridge.memory = memory
    bridge.motion_set = BaselineMotionSet()
    bridge.coordinator = coordinator

    bridge._planned_action_id = 0
    bridge._completed_action_id = None
    bridge._failure_possible_actions = None

    bridge._latest_decision = SimpleNamespace(
        observation=SimpleNamespace(
            place_observation=0,
        ),
        cycle_result=SimpleNamespace(
            planning=SimpleNamespace(
                root_node=SimpleNamespace(
                    place_id=1,
                ),
                available_actions=(
                    0,
                    1,
                ),
            ),
        ),
    )

    return (
        bridge,
        interface,
        coordinator,
    )


def test_action_target_uses_posterior_planning_root():
    """Resolve physical action from the posterior-corrected MCTS root."""
    (
        bridge,
        interface,
        _,
    ) = make_bridge_with_corrected_root()

    target = (
        bridge.resolve_planned_action_target()
    )

    assert target.source_place_id == 1
    assert target.target_place_id == 2

    assert interface.calls == [
        (
            1,
            0,
        )
    ]


def test_failed_action_replans_from_corrected_root():
    """Failure recovery must also retain the corrected planning root."""
    (
        bridge,
        _,
        coordinator,
    ) = make_bridge_with_corrected_root()

    bridge._planned_action_id = None
    bridge._failure_possible_actions = [
        0,
        1,
    ]

    planning = (
        bridge.replan_after_failed_action()
    )

    assert planning.selected_action == 1

    assert coordinator.plan_calls == [
        (
            1,
            (
                0,
                1,
            ),
        )
    ]


def make_fake_navigation_node():
    """Create only the state required by posterior correction."""
    node = SimpleNamespace()

    node._place_memory = (
        BaselinePlaceMemory()
    )

    node._place_memory.resolve_place(
        Point2D(
            0.0,
            0.0,
        )
    )

    node._place_memory.resolve_place(
        Point2D(
            2.0,
            3.0,
        )
    )

    node._internal_cognitive_tracker = (
        BaselineOdomTracker()
    )

    node._internal_cognitive_state = (
        node._internal_cognitive_tracker.reset(
            position=Point2D(
                0.0,
                0.0,
            ),
            travel_heading_rad=1.25,
        )
    )

    node._internal_cognitive_place_id = 0

    return node


def test_confident_posterior_resets_internal_cognitive_pose():
    """A confident posterior snaps internal cognition to that place."""
    node = make_fake_navigation_node()

    decision = SimpleNamespace(
        cycle_result=SimpleNamespace(
            posterior_place_id=1,
        )
    )

    changed = (
        NavigationNode
        ._apply_posterior_cognitive_correction(
            node,
            decision,
        )
    )

    assert changed

    assert (
        node._internal_cognitive_place_id
        == 1
    )

    assert (
        node._internal_cognitive_state.position
        == Point2D(
            2.0,
            3.0,
        )
    )

    assert (
        node._internal_cognitive_state
        .travel_heading_rad
        == pytest.approx(
            0.0
        )
    )


def test_uncertain_posterior_keeps_predicted_cognitive_pose():
    """A -1 posterior result leaves the predicted pose unchanged."""
    node = make_fake_navigation_node()

    before_state = (
        node._internal_cognitive_state
    )

    decision = SimpleNamespace(
        cycle_result=SimpleNamespace(
            posterior_place_id=-1,
        )
    )

    changed = (
        NavigationNode
        ._apply_posterior_cognitive_correction(
            node,
            decision,
        )
    )

    assert not changed

    assert (
        node._internal_cognitive_state
        == before_state
    )

    assert (
        node._internal_cognitive_place_id
        == 0
    )
