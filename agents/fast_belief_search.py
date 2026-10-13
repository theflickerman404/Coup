"""
Fast belief-search Coup agent.

This is the only class that needs to interact with the real Game.
"""

from __future__ import annotations

import random

from agents.fast_coup_search import (
    FastCoupSearch,
    FastSearchConfig,
)
from agents.fast_determinization import (
    FastDeterminizer,
)
from agents.probabilistic_belief import (
    ProbabilisticBeliefTracker,
)
from agents.belief_search import (
    InfluenceCoinEvaluation,
    AggressiveInfluenceEvaluation,
)
from src.cards import Role
from src.player import Player


class FastBeliefSearchAgent(Player):
    """
    Belief-based determinized-search Coup agent.

    Architecture:

        Game
          |
          +-- probabilistic beliefs
          |
          +-- top-probability determinizations
          |
          +-- posterior samples
          |
          +-- lightweight CoupSearchState
          |
          +-- iterative-deepening alpha-beta/PVS
          |
          +-- weighted root-action aggregation
    """

    def __init__(
        self,
        player_id: str,
        name: str | None = None,
        evaluation=None,
        search_config: FastSearchConfig | None = None,
        seed: int | None = None,
        print_turns: bool = True,
    ):
        super().__init__(
            player_id,
            name or player_id,
            print_turns=print_turns,
        )

        self.belief_tracker = (
            ProbabilisticBeliefTracker(
                player_id
            )
        )

        self.evaluation = (
            evaluation
            if evaluation is not None
            else InfluenceCoinEvaluation()
        )

        self.search_config = (
            search_config
            if search_config is not None
            else FastSearchConfig()
        )

        self.rng = random.Random(seed)

        # Reuse the same search object across all determinizations.
        #
        # This is important because:
        #   - history heuristic survives;
        #   - killer moves survive;
        #   - useful TT entries survive.
        self.searcher = FastCoupSearch(
            agent=self,
            evaluation=self.evaluation,
            config=self.search_config,
        )

        self.last_action_scores = {}

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

        if kind in {
            "initial_hand",
            "exchange",
        }:
            if context.get("player_id") == self.id:
                self.belief_tracker.sync_own_cards(
                    [
                        card.role
                        for card in self.cards
                        if not card.revealed
                    ]
                )

    # ------------------------------------------------------------------
    # Main action
    # ------------------------------------------------------------------

    def choose_action(self, game):
        if game.current.id != self.id:
            raise ValueError(
                "It is not this agent's turn."
            )

        # Generate legal root actions once.
        root_actions = self._root_actions(game)

        if not root_actions:
            raise RuntimeError(
                "FastBeliefSearchAgent has no legal actions."
            )

        determinizer = FastDeterminizer(
            game=game,
            tracker=self.belief_tracker,
            perspective_id=self.id,
            top_k=self.search_config.top_completions,
            samples=self.search_config.sampled_completions,
            rng=self.rng,
        )

        worlds = determinizer.generate()

        # Accumulate expected search value for each root action.
        action_values = {
            action: 0.0
            for action in root_actions
        }

        action_weights = {
            action: 0.0
            for action in root_actions
        }

        self.last_action_scores = {}

        # Search high-probability worlds more deeply.
        for world_index, world in enumerate(worlds):
            state = world.state

            available = [
                action
                for action in state.legal_actions(
                    state.current_player
                )
                if action in root_actions
            ]

            if not available:
                continue

            depth = self._depth_for_world(
                world.probability
            )

            time_limit = self._time_for_world(
                world.probability
            )

            result = self.searcher.search(
                state,
                root_actions=available,
                max_depth=depth,
                time_limit=time_limit,
            )

            # IMPORTANT:
            #
            # We want the best move for this world, but we also want to
            # preserve information about the score of alternatives.
            #
            # The simplest robust aggregation is to give the world's
            # probability mass to its best action.
            action_values[result.action] += (
                world.probability
                * result.score
            )

            action_weights[result.action] += (
                world.probability
            )

            self.last_action_scores.setdefault(
                result.action,
                [],
            ).append(
                (
                    world.probability,
                    result.score,
                    result.depth,
                    result.nodes,
                )
            )

        # Fall back safely if all searches timed out before completing.
        if not any(
            weight > 0
            for weight in action_weights.values()
        ):
            selected = root_actions[0]

        else:
            # Expected score over worlds in which this action was selected.
            def final_score(action):
                weight = action_weights[action]

                if weight <= 0:
                    return float("-inf")

                return (
                    action_values[action]
                    / weight
                )

            selected = max(
                root_actions,
                key=final_score,
            )

        action_name, target_index = selected

        target_id = None

        if target_index is not None:
            target_id = game.players[
                target_index
            ].id

        return game.perform_action(
            self.id,
            action_name,
            target_id=target_id,
        )

    # ------------------------------------------------------------------
    # Root actions
    # ------------------------------------------------------------------

    def _root_actions(self, game):
        player_index = next(
            index
            for index, player
            in enumerate(game.players)
            if player.id == self.id
        )

        # Convert the real game into a cheap temporary state only once.
        #
        # This is NOT a search copy. It is only used to enumerate root moves.
        from agents.fast_search_state import (
            CoupSearchState,
        )

        state = CoupSearchState.from_game(
            game
        )

        actions = state.legal_actions(
            player_index
        )

        # Do not allow implausible role claims at the root.
        filtered = []

        for action in actions:
            action_name, _target = action

            role = {
                "tax": Role.DUKE,
                "assassinate": Role.ASSASSIN,
                "exchange": Role.AMBASSADOR,
                "steal": Role.CAPTAIN,
            }.get(action_name)

            if role is not None:
                if not self.claim_is_plausible(
                    game,
                    role,
                ):
                    continue

            filtered.append(action)

        return filtered

    # ------------------------------------------------------------------
    # Belief restriction
    # ------------------------------------------------------------------

    def claim_is_plausible(
        self,
        game,
        role: Role,
    ) -> bool:
        # Truthful claims are always allowed.
        if any(
            not card.revealed
            and card.role == role
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
    # Progressive search budget
    # ------------------------------------------------------------------

    def _depth_for_world(
        self,
        probability: float,
    ) -> int:
        """
        Spend more depth on worlds carrying more probability mass.
        """

        base = self.search_config.max_depth

        if probability >= 0.15:
            return base

        if probability >= 0.07:
            return max(
                1,
                base - 1,
            )

        if probability >= 0.03:
            return max(
                1,
                base - 2,
            )

        return max(
            1,
            base - 4,
        )

    def _time_for_world(
        self,
        probability: float,
    ) -> float:
        base = self.search_config.time_limit

        if probability >= 0.15:
            return base

        if probability >= 0.07:
            return base * 0.85

        if probability >= 0.03:
            return base * 0.65

        return base * 0.40

    # ------------------------------------------------------------------
    # Live responses
    # ------------------------------------------------------------------

    def decide(
        self,
        kind: str,
        context: dict[str, object],
    ) -> object:
        if kind == "challenge":
            return self.should_challenge(
                context
            )

        if kind == "block":
            return self.should_block(
                context
            )

        if kind == "reveal_influence":
            return self.choose_reveal(
                context
            )

        if kind == "exchange":
            return self.choose_exchange(
                context
            )

        raise ValueError(
            f"Unknown decision type: {kind}"
        )

    def should_challenge(
        self,
        context,
    ) -> bool:
        claimant_id = context["claimant_id"]
        role = Role(context["role"])

        if claimant_id == self.id:
            return False

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

        return (
            1.0 - p_believed
            >= self.search_config.challenge_threshold
        )

    def should_block(
        self,
        context,
    ):
        # The real Game provides decision_player_id for the person being
        # asked to block. See src/game.py.
        if context.get(
            "decision_player_id"
        ) != self.id:
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

        # Rule 1: if we can truthfully block, block.
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

        # Rule 2: bluff only when failing to block kills us.
        if (
            action == "assassinate"
            and self.influence == 1
            and context.get("target_id") == self.id
            and Role.CONTESSA in allowed_roles
        ):
            return {
                "blocker_id": self.id,
                "role": Role.CONTESSA.value,
            }

        return None

    def choose_reveal(
        self,
        context,
    ):
        cards = context["cards"]

        values = {
            Role.DUKE.value: 4.0,
            Role.ASSASSIN.value: 5.0,
            Role.CAPTAIN.value: 4.0,
            Role.AMBASSADOR.value: 3.0,
            Role.CONTESSA.value: 3.5,
        }

        return min(
            cards,
            key=lambda card: values[
                card["role"]
            ],
        )["index"]

    def choose_exchange(
        self,
        context,
    ):
        cards = context["cards"]
        keep_count = context["keep_count"]

        values = {
            Role.DUKE.value: 4.0,
            Role.ASSASSIN.value: 5.0,
            Role.CAPTAIN.value: 4.0,
            Role.AMBASSADOR.value: 3.0,
            Role.CONTESSA.value: 3.5,
        }

        ranked = sorted(
            cards,
            key=lambda card: values[
                card["role"]
            ],
            reverse=True,
        )

        return [
            card["index"]
            for card in ranked[:keep_count]
        ]