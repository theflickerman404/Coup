"""
Fast mutable search state for Coup.

This file deliberately does NOT use src.game.Game during search.

The real Game remains authoritative for actual gameplay.  This class is only
used for hypothetical search positions.

The state supports push()/pop(), so search does not allocate or deepcopy a
whole game at every node.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from src.cards import ALL_ROLES, Role


# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------

Action = tuple[str, int | None]


INCOME = "income"
FOREIGN_AID = "foreign_aid"
COUP = "coup"
TAX = "tax"
ASSASSINATE = "assassinate"
EXCHANGE = "exchange"
STEAL = "steal"


ACTION_ROLE = {
    TAX: Role.DUKE,
    ASSASSINATE: Role.ASSASSIN,
    EXCHANGE: Role.AMBASSADOR,
    STEAL: Role.CAPTAIN,
}


BLOCKING_ROLES = {
    FOREIGN_AID: (Role.DUKE,),
    ASSASSINATE: (Role.CONTESSA,),
    STEAL: (Role.AMBASSADOR, Role.CAPTAIN),
}


# ---------------------------------------------------------------------------
# Cards
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class SearchCard:
    role: Role
    revealed: bool = False


@dataclass(slots=True)
class PlayerState:
    player_id: str
    coins: int
    cards: list[SearchCard]

    @property
    def influence(self) -> int:
        return sum(not card.revealed for card in self.cards)

    @property
    def alive(self) -> bool:
        return self.influence > 0

    def has_role(self, role: Role) -> bool:
        return any(
            not card.revealed and card.role == role
            for card in self.cards
        )


# ---------------------------------------------------------------------------
# Undo records
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class UndoRecord:
    """
    Stores exactly the mutable pieces changed by one hypothetical action.

    For simplicity and reliability, card/player lists are shallow-copied
    before an action.  This is still dramatically cheaper than deepcopying
    the entire Game object because SearchCard and PlayerState are tiny and
    there are at most six players / fifteen cards.
    """

    current_player: int
    winner: int | None
    deck: list[SearchCard]
    players: list[PlayerState]


# ---------------------------------------------------------------------------
# Search state
# ---------------------------------------------------------------------------

class CoupSearchState:
    """
    Compact mutable representation of a complete-information Coup state.

    Player indices, rather than player IDs, are used internally for speed.
    """

    def __init__(
        self,
        player_ids: list[str],
        players: list[PlayerState],
        deck: list[SearchCard],
        current_player: int,
    ):
        self.player_ids = player_ids
        self.players = players
        self.deck = deck
        self.current_player = current_player

        self.winner: int | None = None

        self._history: list[UndoRecord] = []

        self.recompute_winner()

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_game(cls, game) -> "CoupSearchState":
        """
        Create a search state from a fully specified Game.

        This should be called ONCE per determinization, not once per search
        node.
        """

        player_ids = [
            player.id
            for player in game.players
        ]

        players = []

        for player in game.players:
            cards = [
                SearchCard(
                    role=card.role,
                    revealed=card.revealed,
                )
                for card in player.cards
            ]

            players.append(
                PlayerState(
                    player_id=player.id,
                    coins=player.coins,
                    cards=cards,
                )
            )

        deck = [
            SearchCard(
                role=card.role,
                revealed=card.revealed,
            )
            for card in game.deck
        ]

        return cls(
            player_ids=player_ids,
            players=players,
            deck=deck,
            current_player=game.current_player,
        )

    # ------------------------------------------------------------------
    # Basic queries
    # ------------------------------------------------------------------

    @property
    def current(self) -> PlayerState:
        return self.players[self.current_player]

    @property
    def alive_players(self) -> list[int]:
        return [
            index
            for index, player in enumerate(self.players)
            if player.alive
        ]

    def player_index(self, player_id: str) -> int:
        return self.player_ids.index(player_id)

    def is_terminal(self) -> bool:
        return len(self.alive_players) <= 1

    def recompute_winner(self) -> None:
        alive = self.alive_players

        if len(alive) == 1:
            self.winner = alive[0]
        else:
            self.winner = None

    # ------------------------------------------------------------------
    # Push / pop
    # ------------------------------------------------------------------

    def push(self, action: Action, policy) -> None:
        """
        Apply one complete hypothetical turn.

        `policy` determines challenges, blocks, card revelation, and exchange
        decisions.
        """

        undo = UndoRecord(
            current_player=self.current_player,
            winner=self.winner,
            deck=[
                SearchCard(c.role, c.revealed)
                for c in self.deck
            ],
            players=[
                PlayerState(
                    player.player_id,
                    player.coins,
                    [
                        SearchCard(c.role, c.revealed)
                        for c in player.cards
                    ],
                )
                for player in self.players
            ],
        )

        self._history.append(undo)

        try:
            self._apply_action(action, policy)
        except Exception:
            self.pop()
            raise

    def pop(self) -> None:
        if not self._history:
            raise RuntimeError("Search-state history is empty.")

        undo = self._history.pop()

        self.current_player = undo.current_player
        self.winner = undo.winner
        self.deck = undo.deck
        self.players = undo.players

    # ------------------------------------------------------------------
    # Action generation
    # ------------------------------------------------------------------

    def legal_actions(
        self,
        player_index: int | None = None,
    ) -> list[Action]:
        if player_index is None:
            player_index = self.current_player

        player = self.players[player_index]

        if not player.alive:
            return []

        # Forced Coup.
        if player.coins >= 10:
            return [
                (COUP, target)
                for target in self.alive_players
                if target != player_index
            ]

        actions: list[Action] = [
            (INCOME, None),
            (FOREIGN_AID, None),
        ]

        if player.coins >= 7:
            actions.extend(
                (COUP, target)
                for target in self.alive_players
                if target != player_index
            )

        # Tax.
        actions.append((TAX, None))

        # Assassin.
        if player.coins >= 3:
            actions.extend(
                (ASSASSINATE, target)
                for target in self.alive_players
                if target != player_index
            )

        # Exchange.
        actions.append((EXCHANGE, None))

        # Steal.
        actions.extend(
            (STEAL, target)
            for target in self.alive_players
            if target != player_index
        )

        return actions

    # ------------------------------------------------------------------
    # Action implementation
    # ------------------------------------------------------------------

    def _apply_action(self, action: Action, policy) -> None:
        actor_index = self.current_player
        actor = self.players[actor_index]

        action_name, target_index = action

        if action_name == INCOME:
            actor.coins += 1

        elif action_name == FOREIGN_AID:
            blocked = self._resolve_block(
                actor_index,
                action_name,
                None,
                policy,
            )

            if not blocked:
                actor.coins += 2

        elif action_name == COUP:
            if actor.coins < 7:
                raise ValueError("Insufficient coins for Coup.")

            actor.coins -= 7

            if target_index is None:
                raise ValueError("Coup requires a target.")

            self._lose_influence(
                target_index,
                policy,
            )

        elif action_name in ACTION_ROLE:
            claimed_role = ACTION_ROLE[action_name]

            challenged = self._resolve_challenge(
                actor_index,
                claimed_role,
                policy,
            )

            if not challenged:
                if action_name == TAX:
                    actor.coins += 3

                elif action_name == ASSASSINATE:
                    if actor.coins < 3:
                        raise ValueError(
                            "Insufficient coins for assassination."
                        )

                    actor.coins -= 3

                    if target_index is not None:
                        blocked = self._resolve_block(
                            actor_index,
                            action_name,
                            target_index,
                            policy,
                        )

                        if not blocked:
                            self._lose_influence(
                                target_index,
                                policy,
                            )

                elif action_name == EXCHANGE:
                    self._resolve_exchange(
                        actor_index,
                        policy,
                    )

                elif action_name == STEAL:
                    if target_index is None:
                        raise ValueError(
                            "Steal requires a target."
                        )

                    blocked = self._resolve_block(
                        actor_index,
                        action_name,
                        target_index,
                        policy,
                    )

                    if not blocked:
                        target = self.players[target_index]
                        stolen = min(2, target.coins)
                        target.coins -= stolen
                        actor.coins += stolen

        self.recompute_winner()

        if self.winner is None:
            self._advance_turn()

    # ------------------------------------------------------------------
    # Challenges
    # ------------------------------------------------------------------

    def _resolve_challenge(
        self,
        claimant: int,
        role: Role,
        policy,
    ) -> bool:
        """
        Return True if the claim is successfully challenged.

        If the claimant actually has the role, the challenger loses influence.
        Otherwise claimant loses influence.
        """

        for offset in range(1, len(self.players) + 1):
            challenger = (
                claimant + offset
            ) % len(self.players)

            if challenger == claimant:
                continue

            if not self.players[challenger].alive:
                continue

            should_challenge = policy.should_challenge(
                self,
                challenger,
                claimant,
                role,
            )

            if not should_challenge:
                continue

            # Challenge succeeds if claimant cannot prove role.
            if not self.players[claimant].has_role(role):
                self._lose_influence(
                    claimant,
                    policy,
                )
                return True

            # Claimant proves role.
            #
            # In the real game the claimed card is returned to the Court and
            # replaced. For search, choose a deterministic replacement from
            # the deck. This preserves the information-changing effect while
            # avoiding random.shuffle.
            self._replace_claimed_card(
                claimant,
                role,
            )

            self._lose_influence(
                challenger,
                policy,
            )

            return False

        return False

    def _replace_claimed_card(
        self,
        claimant: int,
        role: Role,
    ) -> None:
        player = self.players[claimant]

        card_index = next(
            index
            for index, card in enumerate(player.cards)
            if not card.revealed and card.role == role
        )

        returned = player.cards.pop(card_index)
        returned.revealed = False

        self.deck.append(returned)

        if not self.deck:
            raise RuntimeError(
                "Court unexpectedly empty during replacement."
            )

        replacement = self.deck.pop(0)

        player.cards.append(replacement)

    # ------------------------------------------------------------------
    # Blocks
    # ------------------------------------------------------------------

    def _resolve_block(
        self,
        actor: int,
        action: str,
        target: int | None,
        policy,
    ) -> bool:
        if action == FOREIGN_AID:
            eligible = [
                player
                for player in self.alive_players
                if player != actor
            ]
            roles = BLOCKING_ROLES[action]

            for blocker in eligible:
                selection = policy.choose_block(
                    self,
                    blocker,
                    actor,
                    action,
                    target,
                    roles,
                )

                if selection is not None:
                    role = selection

                    if role not in roles:
                        continue

                    # Challenge the block.
                    challenged = self._resolve_challenge(
                        blocker,
                        role,
                        policy,
                    )

                    if not challenged:
                        return True

            return False

        if target is None:
            return False

        roles = BLOCKING_ROLES[action]

        selection = policy.choose_block(
            self,
            target,
            actor,
            action,
            target,
            roles,
        )

        if selection is None:
            return False

        if selection not in roles:
            return False

        challenged = self._resolve_challenge(
            target,
            selection,
            policy,
        )

        return not challenged

    # ------------------------------------------------------------------
    # Influence
    # ------------------------------------------------------------------

    def _lose_influence(
        self,
        player_index: int,
        policy,
    ) -> None:
        player = self.players[player_index]

        active = [
            index
            for index, card in enumerate(player.cards)
            if not card.revealed
        ]

        if not active:
            return

        if len(active) == 1:
            player.cards[active[0]].revealed = True
            self.recompute_winner()
            return

        index = policy.choose_reveal(
            self,
            player_index,
            active,
        )

        if index not in active:
            index = active[0]

        player.cards[index].revealed = True

        self.recompute_winner()

    # ------------------------------------------------------------------
    # Exchange
    # ------------------------------------------------------------------

    def _resolve_exchange(
        self,
        player_index: int,
        policy,
    ) -> None:
        player = self.players[player_index]

        hidden_indices = [
            index
            for index, card in enumerate(player.cards)
            if not card.revealed
        ]

        if len(self.deck) < 2:
            return

        drawn = [
            self.deck.pop(0),
            self.deck.pop(0),
        ]

        candidates = [
            player.cards[index]
            for index in hidden_indices
        ] + drawn

        keep_count = len(hidden_indices)

        kept_indices = policy.choose_exchange(
            self,
            player_index,
            candidates,
            keep_count,
        )

        kept_indices = list(kept_indices)

        if len(kept_indices) != keep_count:
            kept_indices = list(range(keep_count))

        new_hidden = [
            candidates[index]
            for index in kept_indices
            if 0 <= index < len(candidates)
        ]

        if len(new_hidden) != keep_count:
            new_hidden = candidates[:keep_count]

        # Return all non-kept candidate cards to Court.
        returned = [
            card
            for index, card in enumerate(candidates)
            if index not in kept_indices
        ]

        self.deck.extend(returned)

        revealed = [
            card
            for card in player.cards
            if card.revealed
        ]

        player.cards = revealed + new_hidden

    # ------------------------------------------------------------------
    # Turn order
    # ------------------------------------------------------------------

    def _advance_turn(self) -> None:
        start = self.current_player

        while True:
            self.current_player = (
                self.current_player + 1
            ) % len(self.players)

            if self.players[self.current_player].alive:
                return

            if self.current_player == start:
                return