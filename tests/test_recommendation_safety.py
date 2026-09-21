"""The responsible-play path through the recommender.

This is a regression test for a real bug: the flagged-player branch built
adjusted_score from a *filtered* Series, which aligns on index and left every
non-risky game NaN. sort_values then sank the NaNs, so a flagged player
received only the riskiest games — the exact inversion of the rule. It was
silent, because the column looked populated.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from recommendation_engine import (
    build_game_vectors,
    build_player_vectors,
    recommend_for_players,
)


@pytest.fixture(scope="module")
def world(games):
    """A small simulated world: catalogue, sessions, features, risk table."""
    import responsible_play
    from feature_engineering import build_player_features
    from generate_synthetic_data import generate_players, simulate_session
    from schema import DEFAULT_PERSONAS

    rng = np.random.default_rng(3)
    personas = {p.persona_id: p for p in DEFAULT_PERSONAS}
    players = generate_players(DEFAULT_PERSONAS, 8, rng)

    sessions, events = [], []
    for player in players:
        persona = personas[player.persona_id]
        for _ in range(4):
            game = games[rng.integers(0, len(games))]
            session, evs = simulate_session(player, persona, game, rng)
            sessions.append(vars(session) | {"end_reason": session.end_reason.value})
            events.extend(vars(e) for e in evs)

    games_df = pd.DataFrame([vars(g) | {"volatility": g.volatility.value} for g in games])
    raw = {
        "games": games_df,
        "players": pd.DataFrame([vars(p) for p in players]),
        "sessions": pd.DataFrame(sessions),
        "events": pd.DataFrame(events),
    }
    features = build_player_features(raw).fillna({"n_sessions": 0})
    risk = responsible_play.score(features, flag_quantile=0.90)

    game_vecs, theme_cols = build_game_vectors(games_df)
    player_vecs = build_player_vectors(raw["sessions"], games_df, features, theme_cols)
    return {"game_vecs": game_vecs, "player_vecs": player_vecs,
            "risk": risk, "features": features}


def _recommend(world, player_id, n=5):
    return recommend_for_players(
        player_id, world["player_vecs"], world["game_vecs"], world["risk"], top_n=n
    )


def test_flagged_players_get_no_nan_scores(world):
    """The bug itself. Every game must keep a real adjusted_score."""
    flagged = world["risk"].loc[world["risk"].risk_flag, "player_id"]
    assert len(flagged) > 0, "fixture produced no flagged players to test"
    for player_id in flagged:
        recs = _recommend(world, player_id, n=15)
        assert recs["adjusted_score"].notna().all(), player_id


def test_flagged_players_see_fewer_high_volatility_games(world):
    """The rule the downweight is supposed to implement.

    Asserted in aggregate, not per player: the penalty is soft (x0.25 for a
    high-volatility, high-multiplier game), so a single game with a large
    enough similarity lead can still rank first. Before the fix this ratio
    was 1.0 — flagged players got nothing but high-volatility games.
    """
    risk = world["risk"]
    flagged = risk.loc[risk.risk_flag, "player_id"]
    unflagged = risk.loc[~risk.risk_flag, "player_id"]

    def high_vol_share(ids):
        return float(np.mean([_recommend(world, pid)["vol_high"].astype(float).mean()
                              for pid in ids]))

    assert high_vol_share(flagged) < high_vol_share(unflagged)


def test_unflagged_players_are_untouched(world):
    """No adjustment means exactly no adjustment — not 'almost'."""
    unflagged = world["risk"].loc[~world["risk"].risk_flag, "player_id"]
    for player_id in unflagged[:10]:
        recs = _recommend(world, player_id, n=10)
        assert (recs["adjusted_score"] == recs["score"]).all(), player_id
        assert (recs["note"] == "").all(), player_id


def test_ranking_is_sorted_by_adjusted_score(world):
    for player_id in world["risk"]["player_id"][:10]:
        recs = _recommend(world, player_id, n=10)
        assert recs["adjusted_score"].is_monotonic_decreasing, player_id
