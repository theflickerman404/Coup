from agents.fast_search_state import CoupSearchState


class FastInfluenceCoinEvaluation:
    """
    Fast version of InfluenceCoinEvaluation.

    It operates directly on CoupSearchState and therefore does not construct
    Player/Card/Game objects.
    """

    def evaluate_state(
        self,
        state: CoupSearchState,
        perspective_id: str,
    ) -> float:
        me_index = state.player_index(
            perspective_id
        )

        me = state.players[me_index]

        if not me.alive:
            return -1_000_000.0

        if state.winner is not None:
            if state.winner == me_index:
                return 1_000_000.0

            return -1_000_000.0

        score = 0.0

        # Survival is overwhelmingly important.
        score += 1000.0 * me.influence

        # Coins have useful but comparatively small value.
        score += 8.0 * me.coins

        for index, opponent in enumerate(
            state.players
        ):
            if index == me_index:
                continue

            if not opponent.alive:
                continue

            score -= (
                550.0
                * opponent.influence
            )

            score -= (
                3.0
                * opponent.coins
            )

        return score


class FastAggressiveInfluenceEvaluation:
    """
    More aggressive alternative.

    It values eliminating vulnerable opponents relatively highly.
    """

    def evaluate_state(
        self,
        state: CoupSearchState,
        perspective_id: str,
    ) -> float:
        me_index = state.player_index(
            perspective_id
        )

        me = state.players[me_index]

        if not me.alive:
            return -1_000_000.0

        if state.winner is not None:
            return (
                1_000_000.0
                if state.winner == me_index
                else -1_000_000.0
            )

        score = 0.0

        score += (
            1500.0
            * me.influence
        )

        score += (
            4.0
            * me.coins
        )

        for index, opponent in enumerate(
            state.players
        ):
            if index == me_index:
                continue

            if not opponent.alive:
                continue

            score -= (
                800.0
                * opponent.influence
            )

            # A one-influence opponent is an attractive target.
            if opponent.influence == 1:
                score += 250.0

        return score