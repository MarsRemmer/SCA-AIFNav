"""Named navigation modes for SCA-AIFNav."""

from dataclasses import dataclass


EXPLORE = "explore"
GOAL_DIRECT = "goal_direct"
GOAL_BALANCED = "goal_balanced"


@dataclass(frozen=True)
class NavigationModeConfig:
    """Active-inference terms enabled by one navigation mode."""

    name: str
    use_utility: bool
    use_state_information_gain: bool
    use_inductive_inference: bool


_MODE_CONFIGS = {
    EXPLORE: NavigationModeConfig(
        name=EXPLORE,
        use_utility=False,
        use_state_information_gain=True,
        use_inductive_inference=False,
    ),
    GOAL_DIRECT: NavigationModeConfig(
        name=GOAL_DIRECT,
        use_utility=True,
        use_state_information_gain=False,
        use_inductive_inference=True,
    ),
    GOAL_BALANCED: NavigationModeConfig(
        name=GOAL_BALANCED,
        use_utility=True,
        use_state_information_gain=True,
        use_inductive_inference=True,
    ),
}


def navigation_mode_config(
    mode: str,
) -> NavigationModeConfig:
    """Return the fixed active-inference configuration for one mode."""
    if not isinstance(mode, str):
        raise TypeError(
            "navigation mode must be a string"
        )

    try:
        return _MODE_CONFIGS[mode]
    except KeyError as exc:
        raise ValueError(
            "navigation mode must be one of: "
            f"{EXPLORE}, "
            f"{GOAL_DIRECT}, "
            f"{GOAL_BALANCED}"
        ) from exc


def infer_navigation_mode(
    use_utility: bool,
    use_state_information_gain: bool,
    use_inductive_inference: bool,
):
    """Return the named mode matching one set of evaluation switches."""
    for mode, config in _MODE_CONFIGS.items():
        if (
            config.use_utility
            == bool(use_utility)
            and config.use_state_information_gain
            == bool(use_state_information_gain)
            and config.use_inductive_inference
            == bool(use_inductive_inference)
        ):
            return mode

    return None
