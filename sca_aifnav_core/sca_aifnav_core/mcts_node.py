"""Search-tree node primitives for the reference baseline MCTS baseline."""

import math


DEFAULT_UCB_EXPLORATION = 1.41


class SearchTreeNode:
    """One node in the baseline Monte Carlo search tree."""

    def __init__(
        self,
        state_belief,
        place_id: int,
        parent=None,
        action_id=0,
        expected_observation=None,
        initial_reward: float = 0.0,
        possible_actions=None,
    ) -> None:
        self.place_id = place_id

        # The reference implementation also aliases pose_id as the node identifier.
        self.id = place_id

        self.state_belief = state_belief
        self.expected_observation = expected_observation

        self.total_reward = 0.0
        self.visit_count = 0

        self.parent = parent
        self.children = {}
        self.action_id = action_id

        self.possible_actions = possible_actions
        self.untried_actions = None

        self.state_reward = float(
            initial_reward
        )

    def average_reward(self) -> float:
        """Return the mean accumulated reward of this node."""
        if self.visit_count == 0:
            return self.state_reward

        return (
            self.total_reward
            / self.visit_count
        )

    def ucb1_score(
        self,
        c_param: float = DEFAULT_UCB_EXPLORATION,
        use_utility: bool = True,
        use_state_information_gain: bool = True,
    ) -> float:
        """
        Compute reference baseline's UCB1 node-selection score.

        In baseline, the exploitation term is enabled by the utility flag and
        the UCB exploration term is enabled by the state-info-gain flag.
        """
        if self.visit_count == 0:
            return float("inf")

        if self.parent is None:
            parent_visits = self.visit_count
        else:
            parent_visits = (
                self.parent.visit_count
            )

        if parent_visits == 0:
            parent_visits = 1

        exploitation = self.average_reward()

        exploration = (
            c_param
            * math.sqrt(
                math.log(parent_visits)
                / self.visit_count
            )
        )

        score = 0.0

        if use_utility:
            score += exploitation

        if use_state_information_gain:
            score += exploration

        return score

    def is_fully_expanded(self) -> bool:
        """
        Reproduce the baseline's expansion-status check.

        Despite the method name, baseline considers a node expanded once its
        possible-actions field exists and it has at least one child.
        """
        return (
            self.possible_actions is not None
            and len(self.children) > 0
        )

    def has_children(self) -> bool:
        """Return whether the node has any child nodes."""
        return len(self.children) > 0

    def select_best_child_ucb(
        self,
        c_param: float = DEFAULT_UCB_EXPLORATION,
        use_utility: bool = True,
        use_state_information_gain: bool = True,
    ):
        """Return the child with the largest baseline UCB1 score."""
        best_score = -float("inf")
        best_child = None

        for child in self.children.values():
            score = child.ucb1_score(
                c_param=c_param,
                use_utility=use_utility,
                use_state_information_gain=(
                    use_state_information_gain
                ),
            )

            # baseline uses strictly greater-than here. Therefore ties keep
            # the first inserted child.
            if score > best_score:
                best_score = score
                best_child = child

        return best_child

    def child_average_rewards(self):
        """Return child mean rewards in insertion order."""
        return [
            child.average_reward()
            for child in self.children.values()
        ]

    def detach_parent(self) -> None:
        """Turn this node into a root node."""
        self.parent = None
