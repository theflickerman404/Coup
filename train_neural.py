from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch

from src.game import Game
from agents.neural_selfplay import (
    NeuralSelfPlayAgent,
    SharedPolicy,
)


def play_training_game(
    policy: SharedPolicy,
    player_count: int = 4,
    seed: int | None = None,
) -> tuple[list[NeuralSelfPlayAgent], int]:
    """Play one self-play game and return the agents and winner index."""
    rng = random.Random(seed)

    agents = [
        NeuralSelfPlayAgent(
            player_id=f"neural-{i}",
            name=f"Neural-{i}",
            policy=policy,
            seed=rng.randrange(2**31),
            print_turns=False,
            training=True,
        )
        for i in range(player_count)
    ]

    for agent in agents:
        agent.reset_episode()

    game = Game(agents, print_turns=False)

    # Each call executes a complete turn, including all responses.
    turn_limit = 500
    turns = 0

    while game.winner is None:
        if turns >= turn_limit:
            raise RuntimeError(
                "Game exceeded turn limit; inspect the engine "
                "or increase the limit."
            )

        game.current.choose_action(game)
        turns += 1

    winner_index = next(
        i for i, agent in enumerate(agents)
        if agent.id == game.winner.id
    )

    return agents, winner_index


def update_policy(
    policy: SharedPolicy,
    optimizer: torch.optim.Optimizer,
    agents: list[NeuralSelfPlayAgent],
    winner_index: int,
    entropy_coefficient: float = 0.01,
) -> float:
    """Apply one terminal-reward REINFORCE update."""
    player_count = len(agents)
    optimizer.zero_grad()

    losses = []

    for i, agent in enumerate(agents):
        reward = 1.0 if i == winner_index else -1.0 / (
            player_count - 1
        )

        if not agent.log_probs:
            continue

        log_prob_sum = torch.stack(agent.log_probs).sum()
        entropy_sum = (
            torch.stack(agent.entropies).sum()
            if agent.entropies else log_prob_sum * 0.0
        )

        losses.append(
            -reward * log_prob_sum
            - entropy_coefficient * entropy_sum
        )

    if not losses:
        return 0.0

    loss = torch.stack(losses).sum()
    loss.backward()

    # Limit large policy-gradient updates.
    torch.nn.utils.clip_grad_norm_(
        policy.parameters(), max_norm=1.0
    )

    optimizer.step()
    return float(loss.detach().item())


def train(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    policy = SharedPolicy(hidden_size=args.hidden_size)

    checkpoint = Path(args.checkpoint)
    if args.resume and checkpoint.exists():
        policy.load_state_dict(
            torch.load(
                checkpoint,
                map_location="cpu",
                weights_only=True,
            )
        )
        print(f"Loaded checkpoint: {checkpoint}")

    optimizer = torch.optim.Adam(
        policy.parameters(), lr=args.learning_rate
    )

    recent_wins = 0
    recent_games = 0
    recent_losses = []

    for episode in range(1, args.games + 1):
        agents, winner_index = play_training_game(
            policy,
            player_count=args.players,
            seed=args.seed + episode,
        )

        loss = update_policy(
            policy, optimizer, agents, winner_index
        )

        recent_wins += int(winner_index == 0)
        recent_games += 1
        recent_losses.append(loss)

        if episode % args.log_every == 0:
            win_rate = recent_wins / recent_games
            mean_loss = sum(recent_losses) / len(recent_losses)

            print(
                f"Game {episode:>7} | "
                f"seat-0 win rate: {win_rate:.3f} | "
                f"mean loss: {mean_loss:.4f}"
            )

            recent_wins = 0
            recent_games = 0
            recent_losses.clear()

        if episode % args.save_every == 0:
            checkpoint.parent.mkdir(
                parents=True, exist_ok=True
            )

            # Keep a permanent snapshot for this training milestone.
            snapshot = checkpoint.with_name(
                f"{checkpoint.stem}_game_{episode:05d}{checkpoint.suffix}"
            )
            torch.save(policy.state_dict(), snapshot)

            # Also update the latest checkpoint for easy resuming.
            torch.save(policy.state_dict(), checkpoint)

            print(f"Saved snapshot: {snapshot}")
            print(f"Updated latest: {checkpoint}")

    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(policy.state_dict(), checkpoint)
    print(f"Final model saved to {checkpoint}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a neural Coup agent with self-play."
    )

    parser.add_argument("--games", type=int, default=20_000)
    parser.add_argument("--players", type=int, default=4)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/coup_policy.pt",
    )
    parser.add_argument("--resume", action="store_true")

    args = parser.parse_args()

    if not 2 <= args.players <= 6:
        parser.error("--players must be between 2 and 6")
    if args.games < 1:
        parser.error("--games must be positive")
    if args.log_every < 1 or args.save_every < 1:
        parser.error("logging and saving intervals must be positive")

    return args


if __name__ == "__main__":
    train(parse_args())