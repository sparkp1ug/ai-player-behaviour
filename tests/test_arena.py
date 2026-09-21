"""The claim the README makes: the bandit beats its baselines.

Kept small enough to run in CI. The full six-seed comparison lives in
`python src/play_loop.py`; this asserts the direction of the effect, not its
exact size.
"""

from __future__ import annotations

import numpy as np
import pytest

from play_loop import run_policy
from schema import DEFAULT_PERSONAS

SEEDS = (7, 11, 23)
ROUNDS = 30
AGENTS_PER_PERSONA = 8


@pytest.fixture(scope="module")
def arena(games):
    themes = sorted({g.theme for g in games})
    out = {}
    for policy in ("random", "static", "adaptive"):
        out[policy] = [
            run_policy(policy, games, themes, AGENTS_PER_PERSONA, ROUNDS,
                       seed, taste_shift_round=None, collect_events=False)
            for seed in SEEDS
        ]
    return out


def _mean_satisfaction(runs):
    return np.array([r["recommendations"]["satisfaction"].mean() for r in runs])


def test_adaptive_beats_random_on_every_seed(arena):
    """Paired by seed: identical agents and machine seed, only the policy
    differs, so the difference is caused by the policy.

    Safe to assert per seed: measured over 20 consecutive seeds at this
    configuration, the difference was positive on 20/20, minimum +0.0085.
    """
    diff = _mean_satisfaction(arena["adaptive"]) - _mean_satisfaction(arena["random"])
    assert (diff > 0).all(), f"per-seed differences: {diff}"


def test_adaptive_beats_static_on_average(arena):
    """Weaker assertion than the random comparison, deliberately.

    The static cosine engine is a genuinely decent cold-start rule. Over the
    same 20 seeds the difference averaged +0.017 but was negative on one of
    them (-0.005), so a per-seed assertion here would buy a flaky test rather
    than more confidence. The mean over three seeds is the honest claim.
    """
    diff = _mean_satisfaction(arena["adaptive"]) - _mean_satisfaction(arena["static"])
    assert diff.mean() > 0, f"per-seed differences: {diff}"


def test_agents_follow_the_adaptive_policy_more(arena):
    accept = {p: np.mean([r["recommendations"]["accepted"].mean() for r in runs])
              for p, runs in arena.items()}
    assert accept["adaptive"] > accept["random"]


def test_safety_layer_holds_for_flagged_players(arena):
    """Not a learned behaviour — a hard constraint, so this is exact.

    The bandit never recommends a high-volatility game to a risk-flagged
    player. Random does it about a third of the time, which is what the
    constraint is there to prevent.
    """
    for runs in (arena["adaptive"],):
        for run in runs:
            recs = run["recommendations"]
            flagged = recs[recs["risk_flagged"]]
            if flagged.empty:
                continue
            assert flagged["recommended_high_volatility"].sum() == 0

    random_flagged = [
        r["recommendations"].query("risk_flagged")["recommended_high_volatility"].mean()
        for r in arena["random"]
    ]
    assert np.nanmean(random_flagged) > 0.2, "baseline should be unconstrained"


def test_every_persona_is_represented(arena):
    recs = arena["adaptive"][0]["recommendations"]
    assert set(recs["persona_id"]) == {p.persona_id for p in DEFAULT_PERSONAS}
