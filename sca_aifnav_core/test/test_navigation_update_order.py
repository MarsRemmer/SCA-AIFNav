"""Tests for baseline-compatible navigation update ordering."""

from sca_aifnav_core.baseline_odometry import (
    CognitiveOdomState,
)
from sca_aifnav_core.generative_model import (
    BaselineGenerativeModel,
)
from sca_aifnav_core.motion_primitives import (
    BaselineMotionSet,
)
from sca_aifnav_core.navigation_cycle import (
    BaselineNavigationCoordinator,
)
from sca_aifnav_core.planar_geometry import (
    Point2D,
)
from sca_aifnav_core.preference_state import (
    BaselinePreferenceState,
)
from sca_aifnav_core.spatial_memory import (
    BaselinePlaceMemory,
)


def test_new_observation_expands_C_before_state_inference(
    monkeypatch,
):
    """Prepare A and C before the first real-state inference."""
    model = BaselineGenerativeModel()

    memory = BaselinePlaceMemory(
        influence_radius=0.5
    )

    memory.resolve_place(
        Point2D(0.0, 0.0)
    )

    motion_set = BaselineMotionSet()

    preferences = BaselinePreferenceState(
        model
    )

    coordinator = BaselineNavigationCoordinator(
        model=model,
        memory=memory,
        motion_set=motion_set,
        preferences=preferences,
        num_simulations=1,
    )

    events = []

    original_sync = (
        preferences.sync_dimensions
    )

    original_infer = (
        model.infer_state_belief
    )

    def recorded_sync(
        current_model,
    ):
        events.append(
            (
                "C",
                current_model.sensory_observations,
            )
        )

        return original_sync(
            current_model
        )

    def recorded_infer(
        *args,
        **kwargs,
    ):
        events.append(
            (
                "infer",
                model.sensory_observations,
                len(preferences.sensory),
            )
        )

        return original_infer(
            *args,
            **kwargs,
        )

    monkeypatch.setattr(
        preferences,
        "sync_dimensions",
        recorded_sync,
    )

    monkeypatch.setattr(
        model,
        "infer_state_belief",
        recorded_infer,
    )

    state = CognitiveOdomState(
        position=Point2D(
            0.0,
            0.0,
        ),
        travel_heading_rad=0.0,
    )

    coordinator.step_and_plan(
        state=state,
        sensory_observation=2,
        place_observation=0,
        executed_action_id=12,
        obstacle_distances=[
            float("nan")
        ] * 12,
        action_selection="deterministic",
    )

    first_inference_index = next(
        index
        for index, event in enumerate(events)
        if event[0] == "infer"
    )

    c_before_inference = [
        event
        for event in events[
            :first_inference_index
        ]
        if event[0] == "C"
    ]

    assert c_before_inference

    assert c_before_inference[-1] == (
        "C",
        3,
    )

    first_inference = events[
        first_inference_index
    ]

    assert first_inference == (
        "infer",
        3,
        3,
    )
