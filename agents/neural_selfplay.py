from __future__ import annotations

import itertools
import math
import random
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.distributions import Categorical

from src.game import Game
from src.player import Player


ROLES = ["duke", "assassin", "captain", "ambassador", "contessa"]

ACTIONS = [
    "income",
    "foreign_aid",
    "coup",
    "tax",
    "assassinate",
    "exchange",
    "steal",
]

ROLE_FOR_ACTION = {
    "tax": "duke",
    "assassinate": "assassin",
    "exchange": "ambassador",
    "steal": "captain",
}

ACTION_INDEX = {name: i for i, name in enumerate(ACTIONS)}
ROLE_INDEX = {name: i for i, name in enumerate(ROLES)}

# State: six player slots, each with eight features, plus four
# global features.
STATE_DIM = 6 * 8 + 4

# Option: seven action features, six target features, five role
# features, five decision-type features, and five auxiliary features.
OPTION_DIM = 7 + 6 + 5 + 5 + 5 + 1


def encode_state(game: Game, player: Player) -> list[float]:
    """Encode only information available to this player."""
    view = game.state_for(player.id)
    by_id = {p["id"]: p for p in view["players"]}

    # Put this player first; represent other seats in relative order.
    start = next(
        i for i, p in enumerate(game.players) if p.id == player.id
    )
    ordered = (
        game.players[start:] + game.players[:start]
    )

    features: list[float] = []

    for seat in range(6):
        if seat >= len(ordered):
            features.extend([0.0] * 8)
            continue

        other = ordered[seat]
        info = by_id[other.id]

        features.extend([
            min(info["coins"], 15) / 15.0,
            info["influence"] / 2.0,
            1.0 if other.id == player.id else 0.0,
        ])

        # The game engine returns hidden cards for other players,
        # but exposes this player's own cards and public reveals.
        for role in ROLES:
            count = sum(
                card == role for card in info["cards"]
            )
            features.append(count / 2.0)

    features.extend([
        len(game.alive_players) / 6.0,
        min(player.coins, 15) / 15.0,
        player.influence / 2.0,
        1.0 if game.current.id == player.id else 0.0,
    ])

    assert len(features) == STATE_DIM
    return features


def encode_option(
    *,
    action: str | None = None,
    target_slot: int | None = None,
    role: str | None = None,
    decision_type: str,
    aux_roles: list[str] | None = None,
    aux_value: float = 0.0,
) -> list[float]:
    """Encode one candidate decision for the shared policy."""
    result = [0.0] * OPTION_DIM
    offset = 0

    if action in ACTION_INDEX:
        result[offset + ACTION_INDEX[action]] = 1.0
    offset += 7

    if target_slot is not None and 0 <= target_slot < 6:
        result[offset + target_slot] = 1.0
    offset += 6

    if role in ROLE_INDEX:
        result[offset + ROLE_INDEX[role]] = 1.0
    offset += 5

    types = ["action", "challenge", "block", "reveal", "exchange"]
    result[offset + types.index(decision_type)] = 1.0
    offset += 5

    for r in aux_roles or []:
        if r in ROLE_INDEX:
            result[offset + ROLE_INDEX[r]] += 1.0

    # The last auxiliary feature can encode a binary choice.
    result[-1] = aux_value
    return result


class SharedPolicy(nn.Module):
    """Scores candidate options given a player's observation."""

    def __init__(self, hidden_size: int = 128):
        super().__init__()

        self.scorer = nn.Sequential(
            nn.Linear(STATE_DIM + OPTION_DIM, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )

    def forward(
        self,
        state: torch.Tensor,
        options: torch.Tensor,
    ) -> torch.Tensor:
        # One state, N candidate options.
        expanded_state = state.unsqueeze(0).expand(
            options.shape[0], -1
        )
        combined = torch.cat(
            [expanded_state, options], dim=-1
        )
        return self.scorer(combined).squeeze(-1)


class NeuralSelfPlayAgent(Player):
    """Coup player controlled by a shared trainable policy."""

    def __init__(
        self,
        player_id: str,
        name: str | None = None,
        policy: SharedPolicy | None = None,
        seed: int | None = None,
        print_turns: bool = False,
        training: bool = True,
    ):
        super().__init__(
            player_id,
            name or player_id,
            print_turns=print_turns,
        )

        self.policy = policy or SharedPolicy()
        self.rng = random.Random(seed)
        self.training = training

        self._game: Game | None = None
        self.log_probs: list[torch.Tensor] = []
        self.entropies: list[torch.Tensor] = []

    def reset_episode(self) -> None:
        self.log_probs.clear()
        self.entropies.clear()
        self._game = None

    def _sample(
        self,
        state: list[float],
        option_features: list[list[float]],
    ) -> int:
        """Sample an option and retain its policy-gradient data."""
        device = next(self.policy.parameters()).device

        state_tensor = torch.tensor(
            state, dtype=torch.float32, device=device
        )
        options_tensor = torch.tensor(
            option_features, dtype=torch.float32, device=device
        )

        logits = self.policy(state_tensor, options_tensor)
        distribution = Categorical(logits=logits)

        if self.training:
            choice = distribution.sample()
            self.log_probs.append(distribution.log_prob(choice))
            self.entropies.append(distribution.entropy())
            return int(choice.item())

        return int(torch.argmax(logits).item())

    def _state(self) -> list[float]:
        if self._game is None:
            # A safe fallback for contexts received before the agent
            # has been given the current Game instance.
            return [0.0] * STATE_DIM

        return encode_state(self._game, self)

    def _target_slot(self, target_id: str) -> int:
        assert self._game is not None
        start = next(
            i for i, p in enumerate(self._game.players)
            if p.id == self.id
        )
        ordered = (
            self._game.players[start:]
            + self._game.players[:start]
        )
        return next(
            i for i, p in enumerate(ordered)
            if p.id == target_id
        )

    def choose_action(self, game: Game) -> dict:
        if game.current.id != self.id:
            raise ValueError("It is not this agent's turn.")

        # During self-play, make the current observation available
        # to every neural agent for response decisions.
        for p in game.players:
            if isinstance(p, NeuralSelfPlayAgent):
                p._game = game

        self._game = game
        state = self._state()

        alive_opponents = [
            p for p in game.alive_players if p.id != self.id
        ]

        if self.coins >= 10:
            available = ["coup"]
        else:
            available = ["income", "foreign_aid", "tax", "exchange"]

            if self.coins >= 3:
                available.append("assassinate")
            if self.coins >= 7:
                available.append("coup")

            available.extend(["steal"])

        candidates = []
        features = []

        for action in available:
            targets = (
                alive_opponents
                if action in ("coup", "assassinate", "steal")
                else [None]
            )

            for target in targets:
                target_id = target.id if target is not None else None
                slot = (
                    self._target_slot(target_id)
                    if target_id is not None else None
                )

                candidates.append((action, target_id))
                features.append(encode_option(
                    action=action,
                    target_slot=slot,
                    role=ROLE_FOR_ACTION.get(action),
                    decision_type="action",
                ))

        choice = self._sample(state, features)
        action, target_id = candidates[choice]

        return game.perform_action(
            self.id,
            action,
            target_id=target_id,
        )

    def decide(self, kind: str, context: dict[str, Any]) -> object:
        """Handle challenges, blocks, influence losses, and exchanges."""
        state = self._state()

        if kind == "challenge":
            candidates = [False, True]
            features = [
                encode_option(
                    decision_type="challenge",
                    aux_value=float(answer),
                )
                for answer in candidates
            ]
            return candidates[self._sample(state, features)]

        if kind == "block":
            if self.id not in context.get("eligible_blockers", []):
                return None

            action = context["action"]
            choices: list[tuple[str | None, str | None]] = [(None, None)]

            if action == "foreign_aid":
                choices.append(("duke", self.id))
            else:
                for role in context.get("allowed_roles", []):
                    choices.append((role, self.id))

            features = [
                encode_option(
                    decision_type="block",
                    action=action if action in ACTION_INDEX else None,
                    role=role,
                )
                for role, _ in choices
            ]

            role, blocker_id = choices[self._sample(state, features)]

            if role is None:
                return None

            result = {"blocker_id": blocker_id}
            if action != "foreign_aid":
                result["role"] = role
            return result

        if kind == "reveal_influence":
            cards = context["cards"]
            features = [
                encode_option(
                    decision_type="reveal",
                    role=card["role"],
                )
                for card in cards
            ]
            choice = self._sample(state, features)
            return cards[choice]["index"]

        if kind == "exchange":
            cards = context["cards"]
            keep_count = context["keep_count"]

            # Score complete legal subsets, not individual cards.
            subsets = list(
                itertools.combinations(
                    range(len(cards)), keep_count
                )
            )

            features = []
            for subset in subsets:
                kept_roles = [
                    cards[i]["role"] for i in subset
                ]
                features.append(encode_option(
                    decision_type="exchange",
                    aux_roles=kept_roles,
                ))

            choice = self._sample(state, features)
            return list(subsets[choice])

        raise ValueError(f"Unknown decision type: {kind}")

    def observe(
        self, kind: str, context: dict[str, object]
    ) -> None:
        # A later version can add belief-tracker features here.
        # Public events are never used to reveal hidden opponent cards.
        pass

    def save(self, path: str | Path) -> None:
        torch.save(self.policy.state_dict(), path)

    def load(self, path: str | Path) -> None:
        self.policy.load_state_dict(
            torch.load(path, map_location="cpu", weights_only=True)
        )