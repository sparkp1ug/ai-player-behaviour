"""The refactor that makes a cross-language test possible.

`_spin` used to sample directly from numpy's uniform/exponential/beta
generators. It now takes three uniforms as *inputs* and pushes them through
closed-form quantile functions — inverse-transform sampling.

That change is only safe if the two produce the same distributions, and
"looks about right" is not good enough when every downstream number depends
on it. These tests are the evidence, and they are also what a Rust port has
to reproduce.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import stats

from generate_synthetic_data import (
    _BONUS_BETA_A,
    _BONUS_BETA_B,
    _shape_quantile,
    spin_from_uniforms,
)
from schema import Volatility

N = 100_000
# Assert on the KS statistic against a fixed critical value rather than on a
# p-value. A p-value test fails ~1% of the time at alpha=0.01 by construction;
# this threshold is roughly the 0.1% level and does not flake.
KS_CRITICAL = 1.95 * math.sqrt(2.0 / N)


@pytest.fixture(scope="module")
def rng():
    return np.random.default_rng(20260921)


@pytest.mark.parametrize(
    "volatility, sampler",
    [
        (Volatility.LOW, lambda r, n: r.uniform(0.5, 1.5, n)),
        (Volatility.MEDIUM, lambda r, n: r.uniform(0.2, 1.8, n)),
        (Volatility.HIGH, lambda r, n: r.exponential(1.0, n)),
    ],
)
def test_shape_quantile_matches_the_original_sampler(volatility, sampler, rng):
    """Inverse CDF vs the sampler it replaced, per volatility tier."""
    via_quantile = np.array([_shape_quantile(volatility, u) for u in rng.random(N)])
    via_sampler = sampler(rng, N)

    assert stats.ks_2samp(via_quantile, via_sampler).statistic < KS_CRITICAL


def test_beta_quantile_matches_numpy(rng):
    """Beta(1, b) has closed-form CDF 1-(1-x)^b, hence quantile 1-(1-u)^(1/b).

    This only holds for a == 1, which is asserted at import time in
    generate_synthetic_data.
    """
    via_quantile = 1.0 - (1.0 - rng.random(N)) ** (1.0 / _BONUS_BETA_B)
    via_sampler = rng.beta(_BONUS_BETA_A, _BONUS_BETA_B, N)

    assert stats.ks_2samp(via_quantile, via_sampler).statistic < KS_CRITICAL


@pytest.mark.parametrize("volatility", list(Volatility))
def test_every_shape_has_mean_one(volatility, rng):
    """The property the exact payout_scale solve depends on.

    _payout_scale assumes E[shape] == 1.0 for every tier. If that ever stopped
    being true, RTP calibration would be silently wrong, so it is asserted
    rather than left as a comment.
    """
    draws = np.array([_shape_quantile(volatility, u) for u in rng.random(N)])
    assert draws.mean() == pytest.approx(1.0, abs=5 * draws.std() / math.sqrt(N))


def test_branch_structure(games):
    """The uniforms select branches in the documented order.

    Bonus is tested first and wins outright; the base-game test only sees
    spins that did not trigger a bonus. Getting this nesting wrong is the most
    likely porting error, and it shifts the hit rate by about 30 sigma at the
    sample sizes used in the Rust tests.
    """
    game = games[2]

    # u_bonus below bonus_frequency -> bonus, whatever u_win says
    is_win, _, is_bonus = spin_from_uniforms(game, game.bonus_frequency - 1e-12, 1.0, 0.5)
    assert is_win and is_bonus

    # above it, the base-game test decides
    is_win, _, is_bonus = spin_from_uniforms(
        game, game.bonus_frequency + 1e-12, game.base_win_probability - 1e-12, 0.5
    )
    assert is_win and not is_bonus

    is_win, mult, is_bonus = spin_from_uniforms(
        game, game.bonus_frequency + 1e-12, game.base_win_probability + 1e-12, 0.5
    )
    assert not is_win and not is_bonus and mult == 0.0


def test_losing_spins_pay_nothing(games):
    for game in games:
        is_win, mult, is_bonus = spin_from_uniforms(game, 1.0, 1.0, 0.5)
        assert not is_win and not is_bonus and mult == 0.0
