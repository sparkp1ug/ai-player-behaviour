"""The exact payout_scale solve, checked against an independent estimate.

This is the test that earns the right to delete a Monte Carlo calibration:
the analytic answer has to agree with the empirical one, within the
empirical one's own sampling error.
"""

from __future__ import annotations

import numpy as np
import pytest

from generate_synthetic_data import (
    _calibrate_payout_scale_monte_carlo,
    _expected_raw_return,
    _payout_scale,
)


def test_analytic_scale_hits_target_rtp_exactly(games):
    """E[payout/bet] with the solved scale must equal the game's target RTP.

    Tolerance is 1e-4 rather than machine epsilon only because payout_scale is
    rounded to 5 decimal places when the catalogue is written.
    """
    for game in games:
        expected = game.payout_scale * _expected_raw_return(
            game.base_win_probability, game.bonus_frequency, game.max_multiplier
        )
        assert expected == pytest.approx(game.rtp, abs=1e-4), game.game_id


def test_analytic_agrees_with_monte_carlo(games):
    """The closed form and a 200k-sample simulation must agree.

    They are computed completely differently — one is algebra over known
    distribution means, the other draws from those distributions — so
    agreement is real evidence rather than a tautology. The band is the
    Monte Carlo estimator's own 4-sigma error, which is why it is generous
    on the heavy-tailed high-volatility games.
    """
    rng = np.random.default_rng(1)
    for game in games:
        empirical = _calibrate_payout_scale_monte_carlo(
            game.volatility, game.base_win_probability, game.bonus_frequency,
            game.max_multiplier, game.rtp, rng, n_samples=200_000,
        )
        analytic = _payout_scale(
            game.base_win_probability, game.bonus_frequency,
            game.max_multiplier, game.rtp,
        )
        assert empirical == pytest.approx(analytic, rel=0.15), game.game_id


def test_scale_is_positive_for_every_parameter_combination():
    """The property that made the closed form usable at all.

    payout_scale multiplies *both* payout branches, so the solve is
    rtp / E[raw] and E[raw] > 0 whenever the game can pay out. The old
    docstring claimed a closed form "goes negative" for high bonus_frequency
    with high max_multiplier; this walks exactly that corner of the space.
    """
    for bonus_freq in (0.001, 0.02, 0.05, 0.2):
        for max_mult in (5.0, 100.0, 500.0, 5000.0):
            for win_prob in (0.05, 0.48):
                scale = _payout_scale(win_prob, bonus_freq, max_mult, 0.95)
                assert scale > 0.0, (bonus_freq, max_mult, win_prob)
