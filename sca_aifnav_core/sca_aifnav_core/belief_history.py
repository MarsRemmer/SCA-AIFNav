"""State-belief history for the baseline."""

import numpy as np


class BeliefHistory:
    """Store saved physical-state beliefs in baseline order."""

    def __init__(
        self,
        initial_belief: np.ndarray,
    ) -> None:
        self._entries = []

        self.record(
            initial_belief
        )

    def __len__(self) -> int:
        return len(self._entries)

    @property
    def latest(self) -> np.ndarray:
        """Return a copy of the latest saved belief."""
        return self._entries[-1].copy()

    def record(
        self,
        belief: np.ndarray,
    ) -> None:
        """Append one saved baseline state inference."""
        result = np.asarray(
            belief,
            dtype=float,
        )

        if result.ndim != 1:
            raise ValueError(
                "belief must be one-dimensional"
            )

        if not np.all(np.isfinite(result)):
            raise ValueError(
                "belief must contain finite values"
            )

        self._entries.append(
            result.copy()
        )

    def align_to_states(
        self,
        num_states: int,
    ) -> None:
        """Pad all older beliefs after cognitive model growth."""
        if (
            isinstance(num_states, bool)
            or not isinstance(num_states, int)
        ):
            raise TypeError(
                "num_states must be an integer"
            )

        if num_states <= 0:
            raise ValueError(
                "num_states must be positive"
            )

        aligned = []

        for belief in self._entries:
            if len(belief) > num_states:
                raise ValueError(
                    "history belief exceeds model state count"
                )

            missing = (
                num_states - len(belief)
            )

            if missing > 0:
                belief = np.append(
                    belief,
                    np.zeros(missing),
                )

            aligned.append(
                belief
            )

        self._entries = aligned

    def entries(self):
        """Return independent copies of all saved beliefs."""
        return tuple(
            belief.copy()
            for belief in self._entries
        )
