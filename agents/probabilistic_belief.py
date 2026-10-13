from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable

from agents.belief import BeliefTracker
from src.cards import ALL_ROLES, Role


EPSILON = 1e-6


@dataclass
class BetaBelief:
    """
    Beta(alpha, beta) estimate.

    Used for quantities such as:
        P(opponent challenges a Duke claim)

    The posterior mean is alpha / (alpha + beta).
    """

    alpha: float = 1.0
    beta: float = 1.0

    @property
    def mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    def observe_true(self, weight: float = 1.0) -> None:
        self.alpha += weight

    def observe_false(self, weight: float = 1.0) -> None:
        self.beta += weight


class ProbabilisticBeliefTracker(BeliefTracker):
    """
    Probabilistic extension of the repository's BeliefTracker.

    For every player and role we maintain a soft probability that the player
    currently has at least one copy of that role.

    We additionally maintain:
        - claim counts;
        - evidence against claims;
        - opponent challenge propensities;
        - a second-order estimate of what each opponent believes.

    This is deliberately not a mathematically exact Bayesian posterior:
    Coup's observations are highly dependent and the strategy generating a
    claim is itself endogenous. Instead, this class provides a coherent
    probability distribution suitable for determinization and search.
    """

    def __init__(
        self,
        player_id: str,
        prior_role_probability: float = 0.20,
        claim_strength: float = 0.65,
        contradiction_penalty: float = 0.90,
    ):
        super().__init__(player_id)

        self.prior_role_probability = prior_role_probability
        self.claim_strength = claim_strength
        self.contradiction_penalty = contradiction_penalty

        self._role_score: dict[str, dict[Role, float]] = defaultdict(
            lambda: {role: prior_role_probability for role in ALL_ROLES}
        )

        self._claim_count: dict[str, dict[Role, int]] = defaultdict(
            lambda: defaultdict(int)
        )

        self._disproven_count: dict[str, dict[Role, int]] = defaultdict(
            lambda: defaultdict(int)
        )

        # observer -> role -> Beta(challenge, pass)
        self._challenge_rate: dict[
            str, dict[Role, BetaBelief]
        ] = defaultdict(
            lambda: {
                role: BetaBelief()
                for role in ALL_ROLES
            }
        )

    def observe(self, kind: str, context: dict[str, object]) -> None:
        super().observe(kind, context)

        if kind == "new_game":
            self._role_score.clear()
            self._claim_count.clear()
            self._disproven_count.clear()
            self._challenge_rate.clear()
            return

        if kind == "initial_hand":
            player_id = context["player_id"]
            if isinstance(player_id, str):
                self._role_score[player_id] = {
                    role: 1.0 for role in ALL_ROLES
                }
            return

        if kind == "claim":
            claimant_id = context.get("claimant_id")
            role_value = context.get("role")

            if not isinstance(claimant_id, str):
                return

            role = Role(role_value)

            if context.get("disproven", False):
                self._disproven_count[claimant_id][role] += 1
                self._role_score[claimant_id][role] *= (
                    1.0 - self.contradiction_penalty
                )

            elif context.get("proven", False):
                # A proven claim is strong evidence that the player had the
                # claimed role at that moment.
                self._role_score[claimant_id][role] += 3.0

            else:
                self._claim_count[claimant_id][role] += 1
                self._role_score[claimant_id][role] += self.claim_strength

            return

        if kind == "challenge_decision":
            observer = context.get("challenger_id")
            role_value = context.get("role")
            challenged = context.get("challenged")

            if (
                not isinstance(observer, str)
                or not isinstance(challenged, bool)
            ):
                return

            role = Role(role_value)
            beta = self._challenge_rate[observer][role]

            if challenged:
                beta.observe_true()
            else:
                beta.observe_false()

    # ------------------------------------------------------------------
    # First-order beliefs
    # ------------------------------------------------------------------

    def role_probability(
        self,
        player_id: str,
        role: Role | str,
    ) -> float:
        """
        Estimate P(player currently possesses role).

        This is intentionally a probability of possession rather than a
        probability that a particular physical card has the role.
        """

        role = role if isinstance(role, Role) else Role(role)

        if player_id == self.player_id:
            # The agent's own hand is known exactly.
            roles = self.beliefs.get(player_id, [])
            return 1.0 if role in roles else 0.0

        scores = self._role_score[player_id]

        total = sum(max(EPSILON, x) for x in scores.values())
        role_score = max(EPSILON, scores[role])

        # The normalized value is useful as a soft distribution over roles.
        p = role_score / total

        # Convert "role is the most plausible role" into an approximate
        # probability of possessing it. Two influence cards means possession
        # probabilities should be somewhat broader than a one-card categorical
        # distribution.
        return min(1.0, 1.8 * p)

    def role_distribution(
        self,
        player_id: str,
    ) -> dict[Role, float]:
        raw = {
            role: self.role_probability(player_id, role)
            for role in ALL_ROLES
        }

        total = sum(raw.values())

        if total <= EPSILON:
            return {
                role: 1.0 / len(ALL_ROLES)
                for role in ALL_ROLES
            }

        return {
            role: value / total
            for role, value in raw.items()
        }

    def claim_confidence(
        self,
        player_id: str,
        role: Role | str,
    ) -> float:
        role = role if isinstance(role, Role) else Role(role)

        claims = self._claim_count[player_id][role]
        disproven = self._disproven_count[player_id][role]

        score = self._role_score[player_id][role]

        # Strongly reward repeated/proven claims, strongly penalize
        # disproven ones.
        evidence = score + 0.5 * claims - 2.0 * disproven

        return max(0.0, min(1.0, evidence / (evidence + 2.0)))

    # ------------------------------------------------------------------
    # Second-order beliefs
    # ------------------------------------------------------------------

    def opponent_challenge_probability(
        self,
        observer_id: str,
        role: Role | str,
    ) -> float:
        role = role if isinstance(role, Role) else Role(role)
        return self._challenge_rate[observer_id][role].mean

    def opponent_belief_probability(
        self,
        observer_id: str,
        claimant_id: str,
        role: Role | str,
    ) -> float:
        """
        Estimate:

            P(observer believes claimant has role)

        We cannot directly observe an opponent's private cards. We therefore
        combine:

            1. our public/first-order estimate of claimant's role;
            2. how skeptical this particular observer has historically been.

        The result is deliberately conservative. It is principally intended
        for deciding whether a bluff is socially plausible, not for claiming
        exact Bayesian semantics.
        """

        role = role if isinstance(role, Role) else Role(role)

        public_belief = self.role_probability(claimant_id, role)
        challenge_rate = self.opponent_challenge_probability(
            observer_id,
            role,
        )

        # A skeptical opponent discounts the public claim.
        belief = (
            0.70 * public_belief
            + 0.30 * (1.0 - challenge_rate)
        )

        return max(0.0, min(1.0, belief))

    def average_opponent_belief(
        self,
        claimant_id: str,
        role: Role | str,
        opponents: Iterable[str],
    ) -> float:
        role = role if isinstance(role, Role) else Role(role)

        values = [
            self.opponent_belief_probability(
                observer_id=observer,
                claimant_id=claimant_id,
                role=role,
            )
            for observer in opponents
            if observer != claimant_id
        ]

        if not values:
            return 1.0

        return sum(values) / len(values)

    # ------------------------------------------------------------------
    # Action plausibility
    # ------------------------------------------------------------------

    def action_believability(
        self,
        claimant_id: str,
        role: Role,
        opponent_ids: Iterable[str],
    ) -> float:
        """
        Estimate how plausible a role claim is to the table.

        This is what the search agent uses to decide whether it is allowed
        to bluff a role.
        """

        return self.average_opponent_belief(
            claimant_id,
            role,
            opponent_ids,
        )

    def likely_role(
        self,
        player_id: str,
    ) -> Role:
        distribution = self.role_distribution(player_id)
        return max(distribution, key=distribution.get)