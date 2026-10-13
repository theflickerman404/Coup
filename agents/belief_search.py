from __future__ import annotations

import copy
import math
import random
import time

from dataclasses import dataclass
from typing import Callable, Iterable, Optional

from agents.probabilistic_belief import ProbabilisticBeliefTracker
from src.cards import ALL_ROLES, Card, Role
from src.game import Game
from src.player import Player


INF = float("inf")
WIN_SCORE = 1_000_000.0


Action = tuple[str, str | None]

EvaluationFunction = Callable[[Game, str], float]


# ---------------------------------------------------------------------------
# Search result
# ---------------------------------------------------------------------------

@dataclass
class SearchResult:
    action: Action | None
    score: float
    depth: int
    nodes: int
    completed: bool


@dataclass
class Completion:
    """
    A complete assignment of roles to every currently hidden physical card.

    assignments is keyed by (player_id, card_index), while deck assignments
    use ("__deck__", card_index).
    """

    assignments: dict[tuple[str, int], Role]
    weight: float

    @property
    def normalized_weight(self) -> float:
        return self.weight


# ---------------------------------------------------------------------------
# Hidden-information completion
# ---------------------------------------------------------------------------

class HiddenStateEnumerator:
    """
    Enumerate completions of the current information state.

    Every unrevealed opponent card and every card in the Court receives a
    role, subject to the three-copy constraint.

    If max_completions is None, this is exact.

    If max_completions is an integer, enumeration stops after that many
    highest-priority branches. This is an engineering safeguard: the exact
    completion space grows very rapidly with player count.
    """

    DECK_ID = "__deck__"

    def __init__(
        self,
        game: Game,
        tracker: ProbabilisticBeliefTracker,
        perspective_id: str,
        max_completions: int | None = 256,
    ):
        self.game = game
        self.tracker = tracker
        self.perspective_id = perspective_id
        self.max_completions = max_completions

    def enumerate(self) -> list[Completion]:
        known_counts = {
            role: 0
            for role in ALL_ROLES
        }

        slots: list[tuple[str, int]] = []

        # Public/revealed cards and our own cards are known.
        for player in self.game.players:
            for index, card in enumerate(player.cards):
                if (
                    player.id == self.perspective_id
                    or card.revealed
                ):
                    known_counts[card.role] += 1
                else:
                    slots.append((player.id, index))

        # Court cards are hidden from the agent.
        for index, card in enumerate(self.game.deck):
            slots.append((self.DECK_ID, index))

        remaining = {
            role: 3 - known_counts[role]
            for role in ALL_ROLES
        }

        if any(value < 0 for value in remaining.values()):
            raise ValueError(
                "Information state is inconsistent with the 3-copy Coup deck."
            )

        if sum(remaining.values()) != len(slots):
            raise ValueError(
                "Number of hidden cards does not match remaining role counts."
            )

        # Generate high-probability role ordering for each slot.
        role_options: dict[
            tuple[str, int],
            list[tuple[Role, float]]
        ] = {}

        for slot in slots:
            owner, _ = slot

            if owner == self.DECK_ID:
                total = sum(remaining.values())
                options = [
                    (
                        role,
                        remaining[role] / total
                        if remaining[role] else 0.0,
                    )
                    for role in ALL_ROLES
                ]
            else:
                distribution = self.tracker.role_distribution(owner)

                options = [
                    (
                        role,
                        distribution[role]
                    )
                    for role in ALL_ROLES
                    if remaining[role] > 0
                ]

            options.sort(key=lambda pair: pair[1], reverse=True)
            role_options[slot] = options

        # Slots with fewer plausible choices should be assigned first.
        slots.sort(
            key=lambda slot: len(role_options[slot])
        )

        results: list[Completion] = []

        def recurse(
            index: int,
            counts: dict[Role, int],
            assignments: dict[tuple[str, int], Role],
            weight: float,
        ) -> None:
            if (
                self.max_completions is not None
                and len(results) >= self.max_completions
            ):
                return

            if index == len(slots):
                results.append(
                    Completion(
                        assignments=dict(assignments),
                        weight=weight,
                    )
                )
                return

            slot = slots[index]

            for role, probability in role_options[slot]:
                if counts[role] <= 0:
                    continue

                if probability <= 0.0:
                    continue

                counts[role] -= 1
                assignments[slot] = role

                recurse(
                    index + 1,
                    counts,
                    assignments,
                    weight * probability,
                )

                counts[role] += 1
                del assignments[slot]

        recurse(
            0,
            remaining.copy(),
            {},
            1.0,
        )

        if not results:
            raise RuntimeError(
                "No valid hidden-state completions exist. "
                "The belief state may be inconsistent with the game."
            )

        # Normalize weights.
        total_weight = sum(c.weight for c in results)

        if total_weight <= 0.0:
            uniform = 1.0 / len(results)
            for completion in results:
                completion.weight = uniform
        else:
            for completion in results:
                completion.weight /= total_weight

        return results


def apply_completion(
    game: Game,
    completion: Completion,
) -> Game:
    """
    Return a deep copy of game in which every hidden physical card has been
    assigned the role specified by completion.
    """

    result = copy.deepcopy(game)

    for (owner, index), role in completion.assignments.items():
        if owner == HiddenStateEnumerator.DECK_ID:
            result.deck[index].role = role
        else:
            result_player = next(
                player
                for player in result.players
                if player.id == owner
            )
            result_player.cards[index].role = role

    return result


# ---------------------------------------------------------------------------
# Evaluation functions
# ---------------------------------------------------------------------------

class Evaluation:
    """
    Base class for evaluation functions.

    Override evaluate().

    The terminal score is handled by the search itself so evaluation
    functions only need to describe non-terminal positions.
    """

    def __call__(self, game: Game, perspective_id: str) -> float:
        return self.evaluate(game, perspective_id)

    def evaluate(
        self,
        game: Game,
        perspective_id: str,
    ) -> float:
        raise NotImplementedError

    def maximum_action_swing(
            self,
            game: Game,
            perspective_id: str,
    ) -> float:
        """
        A safe upper bound on the amount by which one legal action can
        increase the evaluation.
        
        Returning infinity disables the delta pruning.
        """
        return float("inf")

class InfluenceCoinEvaluation(Evaluation):
    """
    General-purpose Coup heuristic.

    Priority:
        survival > influence > coins.

    Coin value is deliberately modest because coins are only useful insofar
    as they can eventually produce influence loss.
    """

    def evaluate(
        self,
        game: Game,
        perspective_id: str,
    ) -> float:
        me = next(
            p for p in game.players
            if p.id == perspective_id
        )

        if not me.alive:
            return -WIN_SCORE

        if game.winner is not None:
            return (
                WIN_SCORE
                if game.winner.id == perspective_id
                else -WIN_SCORE
            )

        score = 0.0

        # Very strong survival term.
        score += 1000.0 * me.influence

        # Coins matter, but substantially less.
        score += 8.0 * me.coins

        # Relative position matters.
        for opponent in game.alive_players:
            if opponent.id == perspective_id:
                continue

            score -= 550.0 * opponent.influence
            score -= 3.0 * opponent.coins

        return score


class AggressiveInfluenceEvaluation(Evaluation):
    """
    More aggressive evaluation.

    Particularly useful for testing agents that should prefer assassination,
    coups and high-pressure play over passive accumulation.
    """

    def evaluate(
        self,
        game: Game,
        perspective_id: str,
    ) -> float:
        me = next(
            p for p in game.players
            if p.id == perspective_id
        )

        if not me.alive:
            return -WIN_SCORE

        if game.winner is not None:
            return (
                WIN_SCORE
                if game.winner.id == perspective_id
                else -WIN_SCORE
            )

        score = 0.0

        score += 1500.0 * me.influence
        score += 4.0 * me.coins

        # Being close to eliminating an opponent is valuable.
        for opponent in game.alive_players:
            if opponent.id == perspective_id:
                continue

            score -= 800.0 * opponent.influence

            if opponent.influence == 1:
                score += 250.0

        return score


# ---------------------------------------------------------------------------
# Search configuration
# ---------------------------------------------------------------------------

@dataclass
class SearchConfig:
    # Time allocated to one determinization.
    time_limit: float = 0.075

    # Maximum depth reached through iterative deepening.
    max_depth: int = 8

    # Exact information completions unless limited.
    max_completions: int | None = 128

    # Bluff/action plausibility threshold.
    minimum_action_believability: float = 0.25

    # Challenge threshold:
    # challenge if estimated chance that the claim is false is >= this.
    challenge_threshold: float = 0.55

    # Aspiration-window half width.
    aspiration_window: float = 150.0

    # Use principal-variation search.
    use_pvs: bool = True

    # Enable a conservative delta pruning bound.
    use_delta_pruning: bool = True

    # Only prune if evaluator advertises a safe bound.
    delta_margin: float = 100.0

    # Maximum number of table entries.
    transposition_limit: int = 100_000


# ---------------------------------------------------------------------------
# Search engine
# ---------------------------------------------------------------------------

class CoupSearch:
    """
    Determinized alpha-beta search.

    The tree alternates between:

        MAX:
            our agent

        MIN:
            all opponents

    This is a standard paranoid-search approximation for multiplayer Coup:
    all opponents are treated as jointly minimizing our evaluation.

    Optimizations:
        - iterative deepening;
        - alpha-beta;
        - PVS;
        - aspiration windows;
        - transposition table;
        - move ordering;
        - deterministic simulation;
        - optional delta pruning.
    """

    def __init__(
        self,
        agent: "BeliefSearchAgent",
        config: SearchConfig,
    ):
        self.agent = agent
        self.config = config

        self.deadline = 0.0
        self.nodes = 0

        self.transposition: dict[
            tuple,
            tuple[int, float, float, float, Action | None]
        ] = {}

        self.root_pv: list[Action] = []

    # ------------------------------------------------------------------
    # Public search
    # ------------------------------------------------------------------

    def search(
        self,
        game: Game,
        root_actions: list[Action],
    ) -> SearchResult:
        self.deadline = time.perf_counter() + self.config.time_limit
        self.nodes = 0
        self.transposition.clear()
        self.root_pv = []

        if not root_actions:
            raise ValueError("Search was given no legal root actions.")

        best_action = root_actions[0]
        best_score = -INF
        completed_depth = 0

        # Initial ordering.
        ordered = self.order_moves(
            game,
            root_actions,
        )

        previous_score = 0.0

        for depth in range(1, self.config.max_depth + 1):
            if self.out_of_time():
                break

            alpha = -INF
            beta = INF

            # Aspiration window around previous iteration.
            if depth >= 3 and math.isfinite(previous_score):
                alpha = previous_score - self.config.aspiration_window
                beta = previous_score + self.config.aspiration_window

            try:
                score, action = self.root_search(
                    game,
                    ordered,
                    depth,
                    alpha,
                    beta,
                )

                # Re-search if aspiration failed.
                if score <= alpha or score >= beta:
                    score, action = self.root_search(
                        game,
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

            # PV move first next iteration.
            ordered.remove(action)
            ordered.insert(0, action)

        return SearchResult(
            action=best_action,
            score=best_score,
            depth=completed_depth,
            nodes=self.nodes,
            completed=completed_depth > 0,
        )

    # ------------------------------------------------------------------
    # Root
    # ------------------------------------------------------------------

    def root_search(
        self,
        game: Game,
        actions: list[Action],
        depth: int,
        alpha: float,
        beta: float,
    ) -> tuple[float, Action | None]:
        best_score = -INF
        best_action = None

        ordered = self.order_moves(game, actions)

        for index, action in enumerate(ordered):
            self.check_time()

            child = self.apply_action(game, action)

            if self.config.use_pvs and index > 0:
                score = -self.negamax(
                    child,
                    depth - 1,
                    -alpha - 1.0,
                    -alpha,
                    maximizing=False,
                )

                # If the null-window search finds something interesting,
                # perform a full re-search.
                if alpha < score < beta:
                    score = -self.negamax(
                        child,
                        depth - 1,
                        -beta,
                        -alpha,
                        maximizing=False,
                    )
            else:
                score = -self.negamax(
                    child,
                    depth - 1,
                    -beta,
                    -alpha,
                    maximizing=False,
                )

            if score > best_score:
                best_score = score
                best_action = action

            alpha = max(alpha, score)

            if alpha >= beta:
                break

        return best_score, best_action

    # ------------------------------------------------------------------
    # Recursive alpha-beta/PVS
    # ------------------------------------------------------------------

    def delta_prunable(
            self,
            game: Game,
            alpha: float,
            maximizing: bool,
    ) -> bool:
        if not self.config.use_delta_pruning:
            return False

        bound_fn = getattr(
            self.agent.evaluation,
            "maximum_action_swing",
            None,
        )

        if bound_fn is None:
            return False

        swing = bound_fn(
            game,
            self.agent.id
        )

        if not math.isfinite(swing):
            return False

        stand_pat = self.evaluate(game)

        if maximizing:
            return stand_pat + swing <= alpha
        else:
            return stand_pat - swing >= alpha

    def negamax(
        self,
        game: Game,
        depth: int,
        alpha: float,
        beta: float,
        maximizing: bool,
    ) -> float:
        self.check_time()
        self.nodes += 1

        terminal = self.terminal_value(game)

        if terminal is not None:
            return terminal

        if depth <= 0:
            return self.evaluate(game)

        if self.delta_prunable(game, alpha, maximizing):
            return self.evaluate(game)

        key = self.state_key(game, depth, maximizing)

        cached = self.transposition.get(key)
        if cached is not None:
            cached_depth, cached_score, cached_alpha, cached_beta, _ = cached

            if cached_depth >= depth:
                if cached_alpha <= alpha and cached_beta >= beta:
                    return cached_score

        current_id = game.current.id

        actions = self.agent.search_actions(
            game,
            current_id,
            root=False,
        )

        if not actions:
            return self.evaluate(game)

        actions = self.order_moves(game, actions)

        # We are MAX exactly when the current player is the search agent.
        maximizing = current_id == self.agent.id

        if maximizing:
            value = -INF

            for action in actions:
                self.check_time()

                child = self.apply_action(game, action)

                value = max(
                    value,
                    self.negamax(
                        child,
                        depth - 1,
                        alpha,
                        beta,
                        maximizing=False,
                    ),
                )

                alpha = max(alpha, value)

                if alpha >= beta:
                    break

        else:
            value = INF

            for action in actions:
                self.check_time()

                child = self.apply_action(game, action)

                value = min(
                    value,
                    self.negamax(
                        child,
                        depth - 1,
                        alpha,
                        beta,
                        maximizing=False,
                    ),
                )

                beta = min(beta, value)

                if alpha >= beta:
                    break

        if len(self.transposition) < self.config.transposition_limit:
            self.transposition[key] = (
                depth,
                value,
                alpha,
                beta,
                actions[0] if actions else None,
            )

        return value

    # ------------------------------------------------------------------
    # Move ordering
    # ------------------------------------------------------------------

    def order_moves(
        self,
        game: Game,
        actions: list[Action],
    ) -> list[Action]:
        def priority(action: Action) -> float:
            name, target_id = action

            value = 0.0

            if name == "coup":
                value += 1000.0
            elif name == "assassinate":
                value += 850.0
            elif name == "steal":
                value += 400.0
            elif name == "tax":
                value += 300.0
            elif name == "exchange":
                value += 150.0
            elif name == "foreign_aid":
                value += 80.0
            elif name == "income":
                value += 20.0

            if target_id is not None:
                target = next(
                    (
                        p for p in game.players
                        if p.id == target_id
                    ),
                    None,
                )

                if target is not None:
                    value += 200.0 * (2 - target.influence)
                    value += 10.0 * target.coins

            return value

        return sorted(
            actions,
            key=priority,
            reverse=True,
        )

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------

    def apply_action(
        self,
        game: Game,
        action: Action,
    ) -> Game:
        result = copy.deepcopy(game)

        actor_id = result.current.id
        action_name, target_id = action

        # Search responses are deterministic. This is critical because
        # Game itself uses random.shuffle during successful challenges and
        # exchanges.
        random_state = random.getstate()

        try:
            random.seed(
                hash(self.state_key(result, 0, True))
                & 0xFFFFFFFF
            )

            self._search_game = result

            result.perform_action(
                actor_id,
                action_name,
                target_id=target_id,
                decision_provider=self.decision_provider,
            )

        finally:
            random.setstate(random_state)

        return result

    def decision_provider(
        self,
        kind: str,
        context: dict[str, object],
    ) -> object:
        """
        Deterministic response policy used inside search.

        This is not the root agent's live decision routine. It is the
        response policy assumed for all players inside a deterministic
        information completion.
        """

        # ---------------------------------------------------------------
        # Challenge
        # ---------------------------------------------------------------

        if kind == "challenge":
            challenger_id = context["challenger_id"]
            claimant_id = context["claimant_id"]
            role = Role(context["role"])

            # If the challenger actually possesses the role, challenging
            # is generally irrational.
            challenger = self._player(
                self._search_game,
                challenger_id,
            )

            if any(
                not card.revealed and card.role == role
                for card in challenger.cards
            ):
                return False

            # Use the agent's second-order model.
            p_believed = (
                self.agent.belief_tracker
                .opponent_belief_probability(
                    challenger_id,
                    claimant_id,
                    role,
                )
            )

            return (
                1.0 - p_believed
                >= self.config.challenge_threshold
            )

        # ---------------------------------------------------------------
        # Block
        # ---------------------------------------------------------------

        if kind == "block":
            decision_player_id = context.get(
                "decision_player_id"
            )

            if not isinstance(decision_player_id, str):
                return None

            game = self._search_game
            player = self._player(
                game,
                decision_player_id,
            )

            if not player.alive:
                return None

            action = context["action"]

            allowed = [
                Role(role)
                for role in context["allowed_roles"]
            ]

            available = {
                card.role
                for card in player.cards
                if not card.revealed
            }

            possible = [
                role
                for role in allowed
                if role in available
            ]

            if not possible:
                # For the special survival bluff, permit the root agent to
                # bluff only when it is actually the target and would die.
                if (
                    decision_player_id == self.agent.id
                    and action == "assassinate"
                ):
                    target = self._player(
                        game,
                        context["target_id"],
                    )

                    if (
                        target.id == self.agent.id
                        and target.influence == 1
                    ):
                        return {
                            "blocker_id": self.agent.id,
                            "role": Role.CONTESSA.value,
                        }

                return None

            # The requested strategy says to block whenever possible.
            if action == "foreign_aid":
                return {
                    "blocker_id": decision_player_id,
                }

            return {
                "blocker_id": decision_player_id,
                "role": possible[0].value,
            }

        # ---------------------------------------------------------------
        # Reveal
        # ---------------------------------------------------------------

        if kind == "reveal_influence":
            cards = context["cards"]

            # Preserve the most valuable card according to the evaluation
            # function: reveal the least valuable one.
            if len(cards) == 1:
                return cards[0]["index"]

            return cards[-1]["index"]

        # ---------------------------------------------------------------
        # Exchange
        # ---------------------------------------------------------------

        if kind == "exchange":
            cards = context["cards"]
            keep_count = context["keep_count"]

            scored = []

            role_values = {
                Role.DUKE.value: 4.0,
                Role.ASSASSIN.value: 5.0,
                Role.CAPTAIN.value: 4.0,
                Role.AMBASSADOR.value: 3.0,
                Role.CONTESSA.value: 3.5,
            }

            for card in cards:
                scored.append(
                    (
                        role_values.get(
                            card["role"],
                            0.0,
                        ),
                        card["index"],
                    )
                )

            scored.sort(reverse=True)

            return [
                index
                for _, index in scored[:keep_count]
            ]

        raise ValueError(
            f"Unknown search decision type: {kind}"
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _player(
        self,
        game: Game,
        player_id: str,
    ) -> Player:
        return next(
            p for p in game.players
            if p.id == player_id
        )

    def evaluate(self, game: Game) -> float:
        return self.agent.evaluation(
            game,
            self.agent.id,
        )

    def terminal_value(
        self,
        game: Game,
    ) -> float | None:
        if game.winner is not None:
            return (
                WIN_SCORE
                if game.winner.id == self.agent.id
                else -WIN_SCORE
            )

        me = self._player(game, self.agent.id)

        if not me.alive:
            return -WIN_SCORE

        return None

    def state_key(
        self,
        game: Game,
        depth: int,
        maximizing: bool,
    ) -> tuple:
        players = []

        for player in game.players:
            cards = tuple(
                (
                    card.role.value,
                    card.revealed,
                )
                for card in player.cards
            )

            players.append(
                (
                    player.id,
                    player.coins,
                    cards,
                )
            )

        deck = tuple(
            (
                card.role.value,
                card.revealed,
            )
            for card in game.deck
        )

        return (
            game.current.id,
            tuple(players),
            deck,
            depth,
            maximizing,
        )

    def out_of_time(self) -> bool:
        return time.perf_counter() >= self.deadline

    def check_time(self) -> None:
        if self.out_of_time():
            raise SearchTimeout


class SearchTimeout(Exception):
    pass


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class BeliefSearchAgent(Player):
    """
    Coup agent implementing:

        beliefs
          -> hidden-state completions
          -> deterministic search for each completion
          -> weighted aggregation of root actions
          -> action selection

    Blocks:
        1. whenever the agent actually possesses a blocking role;
        2. otherwise, when not blocking would immediately eliminate it.

    Challenges:
        based on the probabilistic/second-order belief tracker.
    """

    def __init__(
        self,
        player_id: str,
        name: str | None = None,
        evaluation: EvaluationFunction | None = None,
        search_config: SearchConfig | None = None,
        print_turns: bool = True,
    ):
        super().__init__(
            player_id,
            name or player_id,
            print_turns=print_turns,
        )

        self.belief_tracker = ProbabilisticBeliefTracker(
            player_id
        )

        self.evaluation = (
            evaluation
            if evaluation is not None
            else InfluenceCoinEvaluation()
        )

        self.search_config = (
            search_config
            if search_config is not None
            else SearchConfig()
        )

        self.last_search_results: list[
            tuple[Completion, SearchResult]
        ] = []

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    def observe(
        self,
        kind: str,
        context: dict[str, object],
    ) -> None:
        self.belief_tracker.observe(
            kind,
            context,
        )

        # Keep our private hand synchronized with the tracker.
        if kind in {
            "initial_hand",
            "exchange",
            "reveal",
        }:
            player_id = context.get("player_id")

            if player_id == self.id:
                self.belief_tracker.sync_own_cards(
                    [
                        card.role
                        for card in self.cards
                        if not card.revealed
                    ]
                )

    # ------------------------------------------------------------------
    # Live action
    # ------------------------------------------------------------------

    def choose_action(
        self,
        game: Game,
    ) -> dict:
        if game.current.id != self.id:
            raise ValueError(
                "It is not this agent's turn."
            )

        completions = HiddenStateEnumerator(
            game=game,
            tracker=self.belief_tracker,
            perspective_id=self.id,
            max_completions=self.search_config.max_completions,
        ).enumerate()

        root_actions = self.search_actions(
            game,
            self.id,
            root=True,
        )

        if not root_actions:
            raise RuntimeError(
                "BeliefSearchAgent has no legal actions."
            )

        aggregate: dict[Action, float] = {
            action: 0.0
            for action in root_actions
        }

        self.last_search_results.clear()

        for completion in completions:
            deterministic_game = apply_completion(
                game,
                completion,
            )

            # Only permit root actions which are legal in this completion.
            deterministic_actions = self.search_actions(
                deterministic_game,
                self.id,
                root=True,
            )

            allowed = [
                action
                for action in root_actions
                if action in deterministic_actions
            ]

            if not allowed:
                continue

            search = CoupSearch(
                agent=self,
                config=self.search_config,
            )

            # The decision provider needs access to the current deterministic
            # game while resolving Game.perform_action.
            search._search_game = deterministic_game

            result = search.search(
                deterministic_game,
                allowed,
            )

            if result.action is None:
                continue

            aggregate[result.action] += (
                completion.weight * result.score
            )

            self.last_search_results.append(
                (
                    completion,
                    result,
                )
            )

        # Weighted average over completions.
        best_action = max(
            aggregate,
            key=aggregate.get,
        )

        action_name, target_id = best_action

        return game.perform_action(
            self.id,
            action_name,
            target_id=target_id,
        )

    # ------------------------------------------------------------------
    # Legal action generation
    # ------------------------------------------------------------------

    def search_actions(
        self,
        game: Game,
        player_id: str,
        root: bool = False,
    ) -> list[Action]:
        player = next(
            p for p in game.players
            if p.id == player_id
        )

        if not player.alive:
            return []

        actions: list[Action] = []

        # 10 coins forces Coup.
        if player.coins >= 10:
            return [
                ("coup", target.id)
                for target in game.alive_players
                if target.id != player.id
            ]

        # All players may Income / Foreign Aid.
        actions.append(("income", None))
        actions.append(("foreign_aid", None))

        # Coup.
        if player.coins >= 7:
            actions.extend(
                (
                    "coup",
                    target.id,
                )
                for target in game.alive_players
                if target.id != player.id
            )

        # Character actions.
        role_actions = {
            Role.DUKE: "tax",
            Role.ASSASSIN: "assassinate",
            Role.AMBASSADOR: "exchange",
            Role.CAPTAIN: "steal",
        }

        for role, action_name in role_actions.items():
            if action_name == "assassinate" and player.coins < 3:
                continue

            if action_name in {
                "assassinate",
                "steal",
            }:
                for target in game.alive_players:
                    if target.id == player.id:
                        continue

                    if root and not self.claim_is_plausible(
                        game,
                        role,
                    ):
                        continue

                    actions.append(
                        (action_name, target.id)
                    )

            else:
                if root and not self.claim_is_plausible(
                    game,
                    role,
                ):
                    continue

                actions.append(
                    (action_name, None)
                )

        return actions

    def claim_is_plausible(
        self,
        game: Game,
        role: Role,
    ) -> bool:
        # Truthful claim.
        if any(
            not card.revealed and card.role == role
            for card in self.cards
        ):
            return True

        opponents = [
            player.id
            for player in game.alive_players
            if player.id != self.id
        ]

        if not opponents:
            return True

        probability = (
            self.belief_tracker.action_believability(
                claimant_id=self.id,
                role=role,
                opponent_ids=opponents,
            )
        )

        return (
            probability
            >= self.search_config.minimum_action_believability
        )

    # ------------------------------------------------------------------
    # Live response policy
    # ------------------------------------------------------------------

    def decide(
        self,
        kind: str,
        context: dict[str, object],
    ) -> object:
        if kind == "challenge":
            return self.should_challenge(context)

        if kind == "block":
            return self.should_block(context)

        if kind == "reveal_influence":
            return self.choose_reveal(context)

        if kind == "exchange":
            return self.choose_exchange(context)

        raise ValueError(
            f"Unknown decision type: {kind}"
        )

    def should_block(
        self,
        context: dict[str, object],
    ) -> object:
        if self.id not in context["eligible_blockers"]:
            return None

        action = context["action"]

        allowed_roles = [
            Role(role)
            for role in context["allowed_roles"]
        ]

        my_roles = {
            card.role
            for card in self.cards
            if not card.revealed
        }

        # Rule 1:
        # If we have a card that allows us to block, block.
        available = [
            role
            for role in allowed_roles
            if role in my_roles
        ]

        if available:
            if action == "foreign_aid":
                return {
                    "blocker_id": self.id,
                }

            return {
                "blocker_id": self.id,
                "role": available[0].value,
            }

        # Rule 2:
        # Bluff a block only when failing to block would immediately
        # eliminate us.
        if action == "assassinate":
            target_id = context.get("target_id")

            if (
                target_id == self.id
                and self.influence == 1
            ):
                return {
                    "blocker_id": self.id,
                    "role": Role.CONTESSA.value,
                }

        return None

    def should_challenge(
        self,
        context: dict[str, object],
    ) -> bool:
        claimant_id = context["claimant_id"]
        role = Role(context["role"])

        if claimant_id == self.id:
            return False

        # If we can prove that the claimant cannot possess the role,
        # challenge immediately.
        if self.belief_tracker.contradicts(
            claimant_id,
            role,
        ):
            return True

        p_believed = (
            self.belief_tracker.opponent_belief_probability(
                observer_id=self.id,
                claimant_id=claimant_id,
                role=role,
            )
        )

        p_false = 1.0 - p_believed

        return (
            p_false
            >= self.search_config.challenge_threshold
        )

    def choose_reveal(
        self,
        context: dict[str, object],
    ) -> int:
        """
        Reveal the least strategically useful card.

        This is deliberately simple; the main search does not depend on
        which of two equivalent influence cards is revealed.
        """

        cards = context["cards"]

        role_value = {
            Role.DUKE.value: 4.0,
            Role.ASSASSIN.value: 5.0,
            Role.CAPTAIN.value: 4.0,
            Role.AMBASSADOR.value: 3.0,
            Role.CONTESSA.value: 3.5,
        }

        return min(
            cards,
            key=lambda card: role_value.get(
                card["role"],
                0.0,
            ),
        )["index"]

    def choose_exchange(
        self,
        context: dict[str, object],
    ) -> list[int]:
        cards = context["cards"]
        keep_count = context["keep_count"]

        role_value = {
            Role.DUKE.value: 4.0,
            Role.ASSASSIN.value: 5.0,
            Role.CAPTAIN.value: 4.0,
            Role.AMBASSADOR.value: 3.0,
            Role.CONTESSA.value: 3.5,
        }

        ranked = sorted(
            cards,
            key=lambda card: role_value.get(
                card["role"],
                0.0,
            ),
            reverse=True,
        )

        return [
            card["index"]
            for card in ranked[:keep_count]
        ]