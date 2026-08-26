"""Tests for first-observation navigation initialization."""

import pytest

from sca_aifnav_core.baseline_odometry import (
    CognitiveOdomState,
)
from sca_aifnav_core.planar_geometry import (
    Point2D,
)
from sca_aifnav_ros.navigation_core_bridge import (
    NavigationCoreBridge,
)
from sca_aifnav_ros.navigation_observation import (
    NavigationObservation,
)


def initial_observation():
    """Create one valid initial navigation observation."""
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


def test_first_observation_has_no_executed_action():
    """Initialization must not fabricate a completed action."""
    bridge = NavigationCoreBridge()

    decision = bridge.process_observation(
        initial_observation()
    )

    assert decision.is_bootstrap is True
    assert decision.executed_action_id is None


def test_first_observation_grows_cognitive_structure():
    """Initial LiDAR evidence should grow the cognitive map."""
    bridge = NavigationCoreBridge()

    before = len(
        bridge.memory
    )

    bridge.process_observation(
        initial_observation()
    )

    assert len(
        bridge.memory
    ) > before


def test_first_observation_plans_directly():
    """Initialization should lead directly to the first MCTS decision."""
    bridge = NavigationCoreBridge()

    decision = bridge.process_observation(
        initial_observation()
    )

    assert decision.cycle_result.learning is None

    assert (
        decision.next_action_id
        == decision.cycle_result.planning.selected_action
    )


def test_second_observation_requires_completed_action():
    """Every observation after initialization needs a real completed action."""
    bridge = NavigationCoreBridge()

    bridge.process_observation(
        initial_observation()
    )

    with pytest.raises(
        RuntimeError,
        match="completed physical action",
    ):
        bridge.process_observation(
            initial_observation()
        )
