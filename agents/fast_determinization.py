"""
Fast hidden-state determinization for Coup.

Produces:
    - a small set of high-probability completions;
    - additional posterior-weighted samples.

This intentionally avoids enumerating the entire combinatorial state space.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Iterable

from src.cards import ALL_ROLES, Role
from agents.fast_search_state import (
    CoupSearchState,
    SearchCard,
    PlayerState,
)


@dataclass(slots=True)
class Determinization:
    state: CoupSearchState
    probability: float


def _copy_card(card) -> SearchCard:
    return SearchCard(
        role=card.role,
        revealed=card.revealed,
    )


def make_determinization(
    game,
    tracker,
    perspective_id: str,
    role_assignments: dict[tuple[str, int], Role],
) -> CoupSearchState:
    """
    Construct a complete-information search state from a role assignment.

    role_assignments keys are:
        (player_id, card_index)

    and:
        ("__deck__", deck_index)
    """

    players: list[PlayerState] = []

    for player in game.players:
        cards = []

        for index, card in enumerate(player.cards):
            role = role_assignments.get(
                (player.id, index),
                card.role,
            )

            cards.append(
                SearchCard(
                    role=role,
                    revealed=card.revealed,
                )
            )

        players.append(
            PlayerState(
                player_id=player.id,
                coins=player.coins,
                cards=cards,
            )
        )

    deck = []

    for index, card in enumerate(game.deck):
        role = role_assignments.get(
            ("__deck__", index),
            card.role,
        )

        deck.append(
            SearchCard(
                role=role,
                revealed=card.revealed,
            )
        )

    return CoupSearchState(
        player_ids=[
            player.id
            for player in game.players
        ],
        players=players,
        deck=deck,
        current_player=game.current_player,
    )


class FastDeterminizer:
    """
    Produces a compact approximation to the posterior over complete hidden
    states.

    The first `top_k` states come from a beam search.

    The remaining states are posterior samples. This gives the search both:
        - concentrated probability mass;
        - coverage of less likely but strategically important worlds.
    """

    DECK_ID = "__deck__"

    def __init__(
        self,
        game,
        tracker,
        perspective_id: str,
        top_k: int = 8,
        samples: int = 24,
        rng: random.Random | None = None,
    ):
        self.game = game
        self.tracker = tracker
        self.perspective_id = perspective_id
        self.top_k = top_k
        self.samples = samples
        self.rng = rng or random.Random()

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def generate(self) -> list[Determinization]:
        slots, remaining = self._hidden_slots()

        if not slots:
            state = make_determinization(
                self.game,
                self.tracker,
                self.perspective_id,
                {},
            )

            return [
                Determinization(
                    state=state,
                    probability=1.0,
                )
            ]

        distributions = {
            slot: self._slot_distribution(
                slot,
                remaining,
            )
            for slot in slots
        }

        beam = self._beam_search(
            slots,
            remaining,
            distributions,
        )

        sampled = self._sample_states(
            slots,
            remaining,
            distributions,
        )

        all_states = beam + sampled

        # Deduplicate.
        unique: dict[tuple, Determinization] = {}

        for determinization in all_states:
            key = self._state_signature(
                determinization.state
            )

            if key in unique:
                unique[key].probability += (
                    determinization.probability
                )
            else:
                unique[key] = determinization

        result = list(unique.values())

        total = sum(
            item.probability
            for item in result
        )

        if total <= 0:
            uniform = 1.0 / len(result)

            for item in result:
                item.probability = uniform

        else:
            for item in result:
                item.probability /= total

        return result

    # ------------------------------------------------------------------
    # Hidden slots
    # ------------------------------------------------------------------

    def _hidden_slots(self):
        known_counts = {
            role: 0
            for role in ALL_ROLES
        }

        slots = []

        for player in self.game.players:
            for index, card in enumerate(player.cards):
                if (
                    player.id == self.perspective_id
                    or card.revealed
                ):
                    known_counts[card.role] += 1
                else:
                    slots.append(
                        (player.id, index)
                    )

        for index, card in enumerate(self.game.deck):
            slots.append(
                (self.DECK_ID, index)
            )

        remaining = {
            role: 3 - known_counts[role]
            for role in ALL_ROLES
        }

        if any(
            count < 0
            for count in remaining.values()
        ):
            raise ValueError(
                "Belief state is inconsistent with the Coup deck."
            )

        if sum(remaining.values()) != len(slots):
            raise ValueError(
                "Hidden-card count is inconsistent."
            )

        return slots, remaining

    # ------------------------------------------------------------------
    # Role distributions
    # ------------------------------------------------------------------

    def _slot_distribution(
        self,
        slot,
        remaining,
    ):
        owner, _index = slot

        if owner == self.DECK_ID:
            total = sum(remaining.values())

            return {
                role: (
                    remaining[role] / total
                    if remaining[role] > 0
                    else 0.0
                )
                for role in ALL_ROLES
            }

        distribution = self.tracker.role_distribution(
            owner
        )

        # Enforce physical-card availability.
        for role in ALL_ROLES:
            if remaining[role] <= 0:
                distribution[role] = 0.0

        total = sum(distribution.values())

        if total <= 0:
            available = [
                role
                for role in ALL_ROLES
                if remaining[role] > 0
            ]

            probability = 1.0 / len(available)

            return {
                role: (
                    probability
                    if role in available
                    else 0.0
                )
                for role in ALL_ROLES
            }

        return {
            role: value / total
            for role, value in distribution.items()
        }

    # ------------------------------------------------------------------
    # Beam search
    # ------------------------------------------------------------------

    def _beam_search(
        self,
        slots,
        remaining,
        distributions,
    ) -> list[Determinization]:
        """
        Beam search through hidden assignments.

        We keep substantially more than top_k intermediate states to avoid
        greedily locking in an early role assignment.
        """

        beam_width = max(
            self.top_k * 8,
            32,
        )

        states = [
            (
                {},
                remaining.copy(),
                1.0,
            )
        ]

        # Most constrained slots first.
        slots = sorted(
            slots,
            key=lambda slot: sum(
                probability > 0
                for probability
                in distributions[slot].values()
            ),
        )

        for slot in slots:
            next_states = []

            for assignments, counts, probability in states:
                options = sorted(
                    distributions[slot].items(),
                    key=lambda item: item[1],
                    reverse=True,
                )

                for role, role_probability in options:
                    if role_probability <= 0:
                        continue

                    if counts[role] <= 0:
                        continue

                    new_assignments = assignments.copy()
                    new_counts = counts.copy()

                    new_assignments[slot] = role
                    new_counts[role] -= 1

                    next_states.append(
                        (
                            new_assignments,
                            new_counts,
                            probability * role_probability,
                        )
                    )

            next_states.sort(
                key=lambda item: item[2],
                reverse=True,
            )

            states = next_states[:beam_width]

        states = states[: self.top_k]

        results = []

        total = sum(
            probability
            for _, _, probability in states
        )

        if total <= 0:
            total = 1.0

        for assignments, _counts, probability in states:
            state = make_determinization(
                self.game,
                self.tracker,
                self.perspective_id,
                assignments,
            )

            results.append(
                Determinization(
                    state=state,
                    probability=probability / total,
                )
            )

        return results

    # ------------------------------------------------------------------
    # Posterior sampling
    # ------------------------------------------------------------------

    def _sample_states(
        self,
        slots,
        remaining,
        distributions,
    ):
        results = []

        for _ in range(self.samples):
            assignments = {}
            counts = remaining.copy()

            # Randomize slot ordering slightly so that samples do not all
            # make identical early assignments.
            ordered_slots = list(slots)
            self.rng.shuffle(ordered_slots)

            probability = 1.0

            for slot in ordered_slots:
                options = [
                    (role, p)
                    for role, p
                    in distributions[slot].items()
                    if p > 0 and counts[role] > 0
                ]

                if not options:
                    break

                roles = [
                    role
                    for role, _ in options
                ]

                weights = [
                    p
                    for _, p in options
                ]

                role = self.rng.choices(
                    roles,
                    weights=weights,
                    k=1,
                )[0]

                p = dict(options)[role]

                probability *= p
                counts[role] -= 1
                assignments[slot] = role

            else:
                state = make_determinization(
                    self.game,
                    self.tracker,
                    self.perspective_id,
                    assignments,
                )

                results.append(
                    Determinization(
                        state=state,
                        probability=probability,
                    )
                )

        return results

    # ------------------------------------------------------------------
    # Hashing
    # ------------------------------------------------------------------

    @staticmethod
    def _state_signature(state):
        players = tuple(
            (
                player.player_id,
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
            (
                card.role.value,
                card.revealed,
            )
            for card in state.deck
        )

        return (
            state.current_player,
            players,
            deck,
        )