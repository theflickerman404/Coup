"""
Fast Coup search.

This module searches CoupSearchState rather than Game.

The search is paranoid minimax:
    - our player maximizes;
    - all opponents minimize our evaluation.

This can later be replaced with Max-N without changing the search-state
architecture.
"""

from __future__ import annotations

import math
import time
from collections import defaultdict
from dataclasses import dataclass

from agents.fast_search_state import (
    Action,
    ASSASSINATE,
    COUP,
    EXCHANGE,
    FOREIGN_AID,
    INCOME,
    STEAL,
    TAX,
    CoupSearchState,
)
from src.cards import Role


INF = float("inf")
WIN_SCORE = 1_000_000.0


class SearchTimeout(Exception):
    pass


@dataclass(slots=True)
class FastSearchConfig:
    time_limit: float = 0.100

    # Depth of the most probable worlds.
    max_depth: int = 10

    # Number of nodes allowed before a hard safety stop.
    max_nodes: int = 2_000_000

    # Hidden-state sampling.
    top_completions: int = 8
    sampled_completions: int = 24

    # Root bluff restriction
    minimum_action_believability: float = 0.20

    # Challenge threshold
    challenge_threshold: float = 0.55

    aspiration_window: float = 150.0

    use_pvs: bool = True
    use_aspiration: bool = True

    transposition_size: int = 250_000

    use_killer_moves: bool = True
    use_history_heuristic: bool = True


@dataclass(slots=True)
class FastSearchResult:
    action: Action
    score: float
    depth: int
    nodes: int


class FastSearchPolicy:
    """
    Supplies opponent challenge/block/reveal/exchange behavior inside the
    hypothetical search tree.

    The policy is intentionally cheap. Search itself decides which ACTION
    an opponent takes; this class decides the response to a challenge/block.
    """

    def __init__(
        self,
        agent,
        belief_tracker=None,
    ):
        self.agent = agent
        self.belief_tracker = belief_tracker

    def should_challenge(
        self,
        state: CoupSearchState,
        challenger: int,
        claimant: int,
        role: Role,
    ) -> bool:
        challenger_player = state.players[challenger]

        # Never challenge if we know the claimant has the role.
        if state.players[claimant].has_role(role):
            return False

        # Our own agent uses its actual belief model.
        if (
            challenger_player.player_id
            == self.agent.id
            and self.belief_tracker is not None
        ):
            p_believed = (
                self.belief_tracker.opponent_belief_probability(
                    observer_id=self.agent.id,
                    claimant_id=state.players[claimant].player_id,
                    role=role,
                )
            )

            return (
                1.0 - p_believed
                >= self.agent.search_config.challenge_threshold
            )

        # Opponents use a cheap rational approximation:
        #
        #   if they know the role is impossible -> challenge;
        #   otherwise challenge with a modest base probability.
        #
        # The complete-information search state tells us the truth, which is
        # deliberately NOT used to make every opponent omniscient.
        if self.belief_tracker is not None:
            claimant_id = state.players[claimant].player_id

            if self.belief_tracker.contradicts(
                claimant_id,
                role,
            ):
                return True

        # Approximate opponent skepticism using role scarcity.
        visible_copies = 0

        for player in state.players:
            for card in player.cards:
                if card.revealed and card.role == role:
                    visible_copies += 1

        # Three visible copies means the claim is impossible.
        if visible_copies >= 3:
            return True

        return False

    def choose_block(
        self,
        state,
        blocker,
        actor,
        action,
        target,
        allowed_roles,
    ):
        player = state.players[blocker]

        # Agent's requested blocking strategy:
        # always block when it possesses a valid role.
        available = [
            role
            for role in allowed_roles
            if player.has_role(role)
        ]

        if available:
            return available[0]

        # If assassination would eliminate the blocker, permit a Contessa
        # bluff. This is exactly the requested "otherwise block if not doing
        # so loses the game" rule.
        if (
            action == ASSASSINATE
            and target == blocker
            and player.influence == 1
            and Role.CONTESSA in allowed_roles
        ):
            return Role.CONTESSA

        return None

    def choose_reveal(
        self,
        state,
        player_index,
        active_indices,
    ):
        player = state.players[player_index]

        # Preserve higher-value cards.
        values = {
            Role.DUKE: 4.0,
            Role.ASSASSIN: 5.0,
            Role.CAPTAIN: 4.0,
            Role.AMBASSADOR: 3.0,
            Role.CONTESSA: 3.5,
        }

        return min(
            active_indices,
            key=lambda index: values[
                player.cards[index].role
            ],
        )

    def choose_exchange(
        self,
        state,
        player_index,
        candidates,
        keep_count,
    ):
        values = {
            Role.DUKE: 4.0,
            Role.ASSASSIN: 5.0,
            Role.CAPTAIN: 4.0,
            Role.AMBASSADOR: 3.0,
            Role.CONTESSA: 3.5,
        }

        ranked = sorted(
            range(len(candidates)),
            key=lambda index: values[
                candidates[index].role
            ],
            reverse=True,
        )

        return ranked[:keep_count]


class FastCoupSearch:
    """
    High-performance deterministic Coup search.

    A single FastCoupSearch instance should preferably be reused across
    determinizations so that its history heuristic and transposition table
    improve move ordering over time.
    """

    EXACT = 0
    LOWERBOUND = 1
    UPPERBOUND = 2

    def __init__(
        self,
        agent,
        evaluation,
        config: FastSearchConfig | None = None,
    ):
        self.agent = agent
        self.evaluation = evaluation
        self.config = config or FastSearchConfig()

        self.policy = FastSearchPolicy(
            agent,
            getattr(agent, "belief_tracker", None),
        )

        self.deadline = 0.0
        self.nodes = 0

        self.tt: dict[
            tuple,
            tuple[int, float, int, Action | None]
        ] = {}

        self.history: defaultdict[
            Action,
            float
        ] = defaultdict(float)

        self.killers: defaultdict[
            int,
            list[Action]
        ] = defaultdict(list)

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def search(
        self,
        state: CoupSearchState,
        root_actions: list[Action] | None = None,
        max_depth: int | None = None,
        time_limit: float | None = None,
    ) -> FastSearchResult:
        if root_actions is None:
            root_actions = self.legal_actions(
                state,
                state.current_player,
            )

        if not root_actions:
            raise ValueError(
                "Cannot search a position with no legal actions."
            )

        max_depth = (
            max_depth
            if max_depth is not None
            else self.config.max_depth
        )

        time_limit = (
            time_limit
            if time_limit is not None
            else self.config.time_limit
        )

        self.deadline = (
            time.perf_counter()
            + time_limit
        )

        self.nodes = 0

        best_action = root_actions[0]
        best_score = -INF
        completed_depth = 0

        ordered = self.order_moves(
            state,
            root_actions,
            depth=0,
            pv_move=None,
        )

        previous_score = 0.0

        for depth in range(1, max_depth + 1):
            try:
                alpha = -INF
                beta = INF

                if (
                    self.config.use_aspiration
                    and depth >= 3
                    and math.isfinite(previous_score)
                ):
                    alpha = (
                        previous_score
                        - self.config.aspiration_window
                    )
                    beta = (
                        previous_score
                        + self.config.aspiration_window
                    )

                score, action = self._root_search(
                    state,
                    ordered,
                    depth,
                    alpha,
                    beta,
                )

                # Aspiration failure.
                if (
                    score <= alpha
                    or score >= beta
                ):
                    score, action = self._root_search(
                        state,
                        ordered,
                        depth,
                        -INF,
                        INF,
                    )

            except SearchTimeout:
                break

            if action is None:
                break

            best_action = action
            best_score = score
            previous_score = score
            completed_depth = depth

            # PV move gets first priority in the next iteration.
            ordered = [
                action
            ] + [
                other
                for other in ordered
                if other != action
            ]

        return FastSearchResult(
            action=best_action,
            score=best_score,
            depth=completed_depth,
            nodes=self.nodes,
        )

    # ------------------------------------------------------------------
    # Root search
    # ------------------------------------------------------------------

    def _root_search(
        self,
        state,
        actions,
        depth,
        alpha,
        beta,
    ):
        best_action = None
        best_score = -INF

        actions = self.order_moves(
            state,
            actions,
            depth=depth,
            pv_move=None,
        )

        for index, action in enumerate(actions):
            self.check_time()

            state.push(
                action,
                self.policy,
            )

            try:
                if (
                    self.config.use_pvs
                    and index > 0
                ):
                    score = self._search(
                        state,
                        depth - 1,
                        alpha,
                        alpha + 1.0,
                    )

                    if alpha < score < beta:
                        score = self._search(
                            state,
                            depth - 1,
                            alpha,
                            beta,
                        )
                else:
                    score = self._search(
                        state,
                        depth - 1,
                        alpha,
                        beta,
                    )
            finally:
                state.pop()

            if score > best_score:
                best_score = score
                best_action = action

            alpha = max(alpha, score)

            if alpha >= beta:
                self.record_cutoff(
                    action,
                    depth,
                )
                break

        return best_score, best_action

    # ------------------------------------------------------------------
    # Alpha-beta
    # ------------------------------------------------------------------

    def _search(
        self,
        state,
        depth,
        alpha,
        beta,
    ):
        self.check_time()
        self.nodes += 1

        terminal = self.terminal_value(state)

        if terminal is not None:
            return terminal

        if depth <= 0:
            return self.evaluate(state)

        maximizing = (
            state.current_player
            == state.player_index(self.agent.id)
        )

        original_alpha = alpha
        original_beta = beta

        key = self.state_key(
            state,
            depth,
        )

        cached = self.tt.get(key)

        if cached is not None:
            cached_depth, score, flag, move = cached

            if cached_depth >= depth:
                if flag == self.EXACT:
                    return score

                if flag == self.LOWERBOUND:
                    alpha = max(alpha, score)

                elif flag == self.UPPERBOUND:
                    beta = min(beta, score)

                if alpha >= beta:
                    return score

        actions = self.legal_actions(
            state,
            state.current_player,
        )

        if not actions:
            return self.evaluate(state)

        pv_move = (
            cached[3]
            if cached is not None
            else None
        )

        actions = self.order_moves(
            state,
            actions,
            depth,
            pv_move,
        )

        if maximizing:
            value = -INF

            for action in actions:
                self.check_time()

                state.push(
                    action,
                    self.policy,
                )

                try:
                    score = self._search(
                        state,
                        depth - 1,
                        alpha,
                        beta,
                    )
                finally:
                    state.pop()

                value = max(value, score)
                alpha = max(alpha, value)

                if alpha >= beta:
                    self.record_cutoff(
                        action,
                        depth,
                    )
                    break

        else:
            value = INF

            for action in actions:
                self.check_time()

                state.push(
                    action,
                    self.policy,
                )

                try:
                    score = self._search(
                        state,
                        depth - 1,
                        alpha,
                        beta,
                    )
                finally:
                    state.pop()

                value = min(value, score)
                beta = min(beta, value)

                if alpha >= beta:
                    self.record_cutoff(
                        action,
                        depth,
                    )
                    break

        if value <= original_alpha:
            flag = self.UPPERBOUND
        elif value >= original_beta:
            flag = self.LOWERBOUND
        else:
            flag = self.EXACT

        if len(self.tt) < self.config.transposition_size:
            self.tt[key] = (
                depth,
                value,
                flag,
                actions[0] if actions else None,
            )

        return value

    # ------------------------------------------------------------------
    # Move ordering
    # ------------------------------------------------------------------

    def order_moves(
        self,
        state,
        actions,
        depth,
        pv_move,
    ):
        killer_moves = self.killers.get(
            depth,
            [],
        )

        def score(action):
            result = 0.0

            if action == pv_move:
                result += 1_000_000

            if action in killer_moves:
                result += 500_000

            if self.config.use_history_heuristic:
                result += self.history[action]

            name, target = action

            # Tactical ordering.
            if name == COUP:
                result += 50_000

            elif name == ASSASSINATE:
                result += 40_000

            elif name == STEAL:
                result += 20_000

            elif name == TAX:
                result += 15_000

            elif name == EXCHANGE:
                result += 8_000

            elif name == FOREIGN_AID:
                result += 3_000

            elif name == INCOME:
                result += 1_000

            if target is not None:
                target_player = state.players[target]

                # Prefer actions against vulnerable players.
                result += (
                    5_000
                    * (2 - target_player.influence)
                )

                result += (
                    10
                    * target_player.coins
                )

            return result

        return sorted(
            actions,
            key=score,
            reverse=True,
        )

    def record_cutoff(
        self,
        action,
        depth,
    ):
        if self.config.use_history_heuristic:
            self.history[action] += depth * depth

        if self.config.use_killer_moves:
            killers = self.killers[depth]

            if action not in killers:
                killers.insert(0, action)

                if len(killers) > 2:
                    killers.pop()

    # ------------------------------------------------------------------
    # Legal moves
    # ------------------------------------------------------------------

    @staticmethod
    def legal_actions(
        state,
        player_index,
    ):
        return state.legal_actions(
            player_index
        )

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(self, state):
        evaluator = self.evaluation

        if hasattr(
            evaluator,
            "evaluate_state",
        ):
            return evaluator.evaluate_state(
                state,
                self.agent.id,
            )

        # Backwards-compatible adapter for the evaluation classes from the
        # previous implementation.
        return evaluator(
            state,
            self.agent.id,
        )

    def terminal_value(self, state):
        if state.winner is not None:
            winner_id = state.players[
                state.winner
            ].player_id

            return (
                WIN_SCORE
                if winner_id == self.agent.id
                else -WIN_SCORE
            )

        me = state.players[
            state.player_index(self.agent.id)
        ]

        if not me.alive:
            return -WIN_SCORE

        return None

    # ------------------------------------------------------------------
    # Transposition key
    # ------------------------------------------------------------------

    def state_key(
        self,
        state,
        depth,
    ):
        """
        Compact immutable state key.

        We include card roles because this is a perfect-information
        determinization.
        """

        players = tuple(
            (
                player.coins,
                tuple(
                    (
                        card.role.value,
                        card.revealed,
                    )
                    for card in player.cards
                ),
            )
            for player in state.players
        )

        deck = tuple(
            card.role.value
            for card in state.deck
        )

        return (
            state.current_player,
            players,
            deck,
            depth,
        )

    # ------------------------------------------------------------------
    # Time
    # ------------------------------------------------------------------

    def check_time(self):
        if self.nodes >= self.config.max_nodes:
            raise SearchTimeout

        if time.perf_counter() >= self.deadline:
            raise SearchTimeout