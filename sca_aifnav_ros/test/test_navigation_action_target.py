"""Tests for planned navigation action target resolution."""

from types import SimpleNamespace

import pytest

from sca_aifnav_core.baseline_odometry import (
    CognitiveOdomState,
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


class FakeModelInterface:
    """Provide deterministic action-to-place mappings."""

    def __init__(
        self,
        target_place_id,
    ):
        self.target_place_id = (
            target_place_id
        )
        self.calls = []

    def get_next_place_id(
        self,
        current_place_id,
        action_id,
    ):
        """Return the configured cognitive target place."""
        self.calls.append(
            (
                current_place_id,
                action_id,
            )
        )

        return self.target_place_id


class _FakeHistory:
    """Provide the initialization history interface."""

    def align_to_states(
        self,
        num_states,
    ):
        """Accept model dimensional growth."""
        return None


class _FakeLearning:
    """Provide baseline initialization parameters."""

    robot_dimension = 0.25
    max_lookahead_steps = 8

    def __init__(
        self,
    ):
        self.history = _FakeHistory()


class _FakePreferences:
    """Provide preference synchronization for bootstrap planning."""

    def sync_dimensions(
        self,
        model,
    ):
        """Accept model dimensional changes."""
        return None

    def snapshot(
        self,
    ):
        """Return a minimal preference snapshot."""
        return None


class _FakeModelInterface:
    """Resolve one predetermined cognitive target."""

    def __init__(
        self,
        target_place_id,
    ):
        self.target_place_id = (
            target_place_id
        )

        self.calls = []

    def get_next_place_id(
        self,
        current_place_id,
        action_id,
    ):
        """Return the target configured by the test."""
        self.calls.append(
            (
                current_place_id,
                action_id,
            )
        )

        return self.target_place_id


class FakeCoordinator:
    """Provide deterministic planning for target-resolution tests."""

    def __init__(
        self,
        selected_action,
        target_place_id,
    ):
        self.selected_action = (
            selected_action
        )

        self.learning = (
            _FakeLearning()
        )

        self.preferences = (
            _FakePreferences()
        )

        self.model_interface = (
            _FakeModelInterface(
                target_place_id
            )
        )

    def plan_current(
        self,
        current_place_id=None,
        possible_actions=None,
        action_selection=None,
        rng=None,
    ):
        """Return the action selected by the test."""
        return SimpleNamespace(
            selected_action=(
                self.selected_action
            )
        )

    def step_and_plan(
        self,
        **kwargs,
    ):
        """Return deterministic planning after a learned action."""
        return SimpleNamespace(
            learning=None,
            preferences=None,
            planning=SimpleNamespace(
                selected_action=(
                    self.selected_action
                )
            ),
        )


def observation(
    place_id=0,
):
    """Create one deterministic navigation observation."""
    return NavigationObservation(
        state=CognitiveOdomState(
            position=Point2D(
                0.0,
                0.0,
            ),
            travel_heading_rad=0.0,
        ),
        sensory_observation=0,
        place_observation=place_id,
        obstacle_distances=tuple(
            1.0
            for _ in range(12)
        ),
        odometry_revision=1,
        scan_revision=1,
    )


def test_no_target_exists_before_planning():
    """A new bridge should not expose a physical target."""
    bridge = NavigationCoreBridge(
        coordinator=FakeCoordinator(
            selected_action=0,
            target_place_id=1,
        )
    )

    assert (
        bridge.resolve_planned_action_target()
        is None
    )


def test_directional_action_resolves_exact_mcts_target():
    """Physical execution should use the same place selected by MCTS."""
    memory = BaselinePlaceMemory()

    source_id = memory.resolve_place(
        Point2D(
            0.0,
            0.0,
        )
    )

    target_id = memory.resolve_place(
        Point2D(
            0.60,
            0.16,
        )
    )

    motion_set = BaselineMotionSet()

    coordinator = FakeCoordinator(
        selected_action=0,
        target_place_id=(
            target_id
        ),
    )

    bridge = NavigationCoreBridge(
        memory=memory,
        motion_set=motion_set,
        coordinator=coordinator,
    )

    bridge.process_observation(
        observation(
            place_id=source_id
        )
    )

    target = (
        bridge.resolve_planned_action_target()
    )

    assert target.action_id == 0
    assert target.source_place_id == source_id
    assert target.target_place_id == target_id
    assert target.is_stationary is False

    assert (
        target.target_position
        == Point2D(
            0.60,
            0.16,
        )
    )

    assert (
        coordinator.model_interface.calls
        == [
            (
                source_id,
                0,
            )
        ]
    )


def test_stay_action_targets_current_place():
    """STAY should resolve to the current cognitive place."""
    memory = BaselinePlaceMemory()

    place_id = memory.resolve_place(
        Point2D(
            2.0,
            -1.0,
        )
    )

    coordinator = FakeCoordinator(
        selected_action=12,
        target_place_id=(
            place_id
        ),
    )

    bridge = NavigationCoreBridge(
        memory=memory,
        motion_set=BaselineMotionSet(),
        coordinator=coordinator,
    )

    bridge.process_observation(
        observation(
            place_id=place_id
        )
    )

    target = (
        bridge.resolve_planned_action_target()
    )

    assert target.action_id == 12
    assert target.target_place_id == place_id
    assert target.is_stationary is True

    assert (
        target.target_position
        == Point2D(
            2.0,
            -1.0,
        )
    )


def test_unknown_mcts_target_is_rejected():
    """A planned physical action must resolve to a known place."""
    memory = BaselinePlaceMemory()

    place_id = memory.resolve_place(
        Point2D(
            0.0,
            0.0,
        )
    )

    bridge = NavigationCoreBridge(
        memory=memory,
        motion_set=BaselineMotionSet(),
        coordinator=FakeCoordinator(
            selected_action=3,
            target_place_id=-1,
        ),
    )

    bridge.process_observation(
        observation(
            place_id=place_id
        )
    )

    with pytest.raises(
        RuntimeError,
        match="known cognitive place",
    ):
        bridge.resolve_planned_action_target()


def test_target_disappearing_from_memory_is_rejected():
    """A resolved target ID must still exist in cognitive memory."""
    memory = BaselinePlaceMemory()

    source_id = memory.resolve_place(
        Point2D(
            0.0,
            0.0,
        )
    )

    bridge = NavigationCoreBridge(
        memory=memory,
        motion_set=BaselineMotionSet(),
        coordinator=FakeCoordinator(
            selected_action=0,
            target_place_id=99,
        ),
    )

    bridge.process_observation(
        observation(
            place_id=source_id
        )
    )

    with pytest.raises(
        RuntimeError,
        match="missing",
    ):
        bridge.resolve_planned_action_target()


def test_completed_action_clears_planned_target():
    """A completed action should no longer expose an execution target."""
    memory = BaselinePlaceMemory()

    source_id = memory.resolve_place(
        Point2D(
            0.0,
            0.0,
        )
    )

    target_id = memory.resolve_place(
        Point2D(
            0.60,
            0.16,
        )
    )

    bridge = NavigationCoreBridge(
        memory=memory,
        motion_set=BaselineMotionSet(),
        coordinator=FakeCoordinator(
            selected_action=0,
            target_place_id=(
                target_id
            ),
        ),
    )

    bridge.process_observation(
        observation(
            place_id=source_id
        )
    )

    assert (
        bridge.resolve_planned_action_target()
        is not None
    )

    bridge.record_executed_action(
        0
    )

    assert (
        bridge.resolve_planned_action_target()
        is None
    )
