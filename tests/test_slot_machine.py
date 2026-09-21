"""The live machine must pay what the catalogue claims."""

from __future__ import annotations

import numpy as np
import pytest

from generate_synthetic_data import _expected_raw_return
from slot_machine import SlotMachine, audit


def _analytic_moments(game):
    """E[m] and SD[m] for one spin's payout multiplier, in closed form.

    The second moment needs each tier's E[shape^2]: U(0.5,1.5) -> 13/12,
    U(0.2,1.8) -> 1.2133, Exp(1) -> 2. And E[Beta(1,12)^2] = 2/(13*14).
    """
    s = game.payout_scale
    bf, p, mm = game.bonus_frequency, game.base_win_probability, game.max_multiplier
    second_moment_by_tier = {"low": 13 / 12, "medium": 1.21333333, "high": 2.0}

    mean = s * _expected_raw_return(p, bf, mm)
    second = (bf * (mm * s) ** 2 * 2 / (13 * 14)
              + (1 - bf) * p * s ** 2 * second_moment_by_tier[game.volatility.value])
    return mean, np.sqrt(max(second - mean ** 2, 0.0))


def test_realised_rtp_matches_analytic_expectation(machine, games):
    """Spin each game and check the realised return against its analytic mean.

    The band is the sampling error of the *test*, not a fudge factor: at
    100k spins a high-volatility game has SD ~8-9, so its standard error is
    ~0.028 and a 5-sigma band is +/-0.14. This is a smoke test by nature —
    it proves the machine is not systematically wrong, not that it is exact.
    """
    n = 100_000
    table = audit(machine, n, bet=1.0).set_index("game_id")
    for game in games:
        mean, sd = _analytic_moments(game)
        realised = table.loc[game.game_id, "realised_rtp"]
        assert realised == pytest.approx(mean, abs=5 * sd / np.sqrt(n)), game.game_id


def test_branch_frequencies(machine, games):
    """Hit rate and bonus rate are Bernoulli, so the bands are tight.

    P(win) = bf + (1-bf)*p. Getting that nesting wrong — using p instead of
    (1-bf)*p — shifts the hit rate by about 30 sigma at this sample size, so
    this is the test that catches a mis-ported branch structure.
    """
    n = 200_000
    table = audit(machine, n, bet=1.0).set_index("game_id")
    for game in games:
        bf, p = game.bonus_frequency, game.base_win_probability
        expected_hit = bf + (1 - bf) * p
        se = np.sqrt(expected_hit * (1 - expected_hit) / n)
        assert table.loc[game.game_id, "hit_rate"] == pytest.approx(expected_hit, abs=5 * se)

        se_bonus = np.sqrt(bf * (1 - bf) / n)
        assert table.loc[game.game_id, "bonus_rate"] == pytest.approx(bf, abs=5 * se_bonus)


def test_bet_is_clamped_to_the_game_limits(games):
    machine = SlotMachine(games, seed=3)
    game = games[0]
    assert machine.spin(game.game_id, 0.0).bet == pytest.approx(game.min_bet)
    assert machine.spin(game.game_id, 1e9).bet == pytest.approx(game.max_bet)


def test_spin_result_invariants(machine, games):
    """What must always hold for a single spin.

    Note what is deliberately NOT asserted: that a win always pays something.
    Payouts are rounded to cents, and a non-bonus win can draw a multiplier
    small enough to round to zero — about 0.06% of wins at a bet of 1.00, and
    0.5% at the 0.10 minimum. Real machines round down too. The invariant that
    does hold is the other direction: a payout implies a win.
    """
    from slot_machine import BONUS_SYMBOL

    for game in games:
        for _ in range(500):
            r = machine.spin(game.game_id, 1.0)
            if not r.is_win:
                assert r.payout == 0.0 and r.multiplier == 0.0
                assert len(set(r.reels)) > 1, "a losing spin must not show three of a kind"
            else:
                assert len(set(r.reels)) == 1, "a winning spin shows three of a kind"
            if r.payout > 0:
                assert r.is_win
            if r.is_bonus:
                assert r.is_win and r.reels == (BONUS_SYMBOL,) * 3
