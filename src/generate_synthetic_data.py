"""
Synthetic player-behaviour dataset generator.

Simulates a set of fictional slot games and player personas spinning through
sessions, with realistic-ish bankroll mechanics: bet escalation after losses,
bet decay after wins, bonus triggers, voluntary quitting, and bankroll
depletion. Everything here is synthetic — no real game logic, RTP
certification, or gambling-operator data is used or implied.

Outputs (under --out-dir, default ./data):
    games.csv     one row per fictional slot game
    players.csv   one row per simulated player
    sessions.csv  one row per session, with aggregated behavioural features
    events.csv    one row per spin (the raw sequence, for anyone who wants
                  to build the sequence-model / LSTM version later)

Usage:
    python generate_synthetic_data.py --seed 42 --players-per-persona 30 \
        --sessions-per-player 3 --n-games 15 --out-dir data
"""

from __future__ import annotations

import argparse
import math
import uuid
from pathlib import Path

import numpy as np
import pandas as pd

from schema import (
    DEFAULT_PERSONAS,
    BetEvent,
    EndReason,
    Player,
    PlayerPersona,
    Session,
    SlotGame,
    Volatility,
)

THEMES = [
    "egyptian", "mythology", "fruits", "pirates", "space", "animals",
    "fantasy", "western", "ocean", "asian_treasures", "holiday", "sports",
    "mining", "vikings", "aztec",
]

# Per-spin win probability range for the NON-bonus outcome, by volatility
# tier. Fixed once per game at generation time (see generate_games) rather
# than resampled every spin, so it behaves like an actual game property.
_WIN_PROB_RANGE = {
    Volatility.LOW: (0.40, 0.48),
    Volatility.MEDIUM: (0.25, 0.32),
    Volatility.HIGH: (0.14, 0.20),
}

# Bonus-round payout multiplier, as a fraction of max_multiplier: Beta(1, 12)
# has mean 1/13 ≈ 0.077, i.e. a bonus round typically pays out well under the
# game's headline "up to Nx" figure, with occasional draws close to it — the
# max_multiplier is a rare ceiling, not a typical outcome (as in real slots).
_BONUS_BETA_A, _BONUS_BETA_B = 1.0, 12.0


def _raw_shape_draw(volatility: Volatility, rng: np.random.Generator) -> float:
    """Unscaled non-bonus win-multiplier shape, mean == 1.0 by construction.
    Only the *shape* (variance) differs by tier; _payout_scale rescales it
    per game so the realised mean matches RTP."""
    if volatility is Volatility.LOW:
        return rng.uniform(0.5, 1.5)          # tight — steady small wins
    if volatility is Volatility.MEDIUM:
        return rng.uniform(0.2, 1.8)          # wider spread
    return rng.exponential(scale=1.0)          # heavy right tail


def _expected_raw_return(win_prob: float, bonus_freq: float, max_mult: float) -> float:
    """E[payout / bet] for one spin at unit payout_scale. Exact, not estimated.

    Every branch of _spin is a probability times the mean of a draw:

        E[raw] = bonus_freq * E[Beta(a,b)] * max_mult
               + (1 - bonus_freq) * win_prob * E[shape]

    and both means are known in closed form. E[Beta(1,12)] = 1/13, and every
    tier's shape draw has mean exactly 1.0 by construction — U(0.5,1.5),
    U(0.2,1.8) and Exp(1) all do, which is the whole point of
    _raw_shape_draw only varying the *variance* between tiers. So volatility
    does not appear here at all.
    """
    beta_mean = _BONUS_BETA_A / (_BONUS_BETA_A + _BONUS_BETA_B)
    return bonus_freq * beta_mean * max_mult + (1.0 - bonus_freq) * win_prob


def _payout_scale(win_prob: float, bonus_freq: float, max_mult: float,
                  target_rtp: float) -> float:
    """Solve exactly for the factor that brings E[payout/bet] to target_rtp.

    This replaces a 40,000-sample Monte Carlo calibration that was measurably
    biased: on the shipped catalogue it missed the target RTP by a mean of
    1.68pp and by up to 5.57pp, worst on exactly the high-volatility games
    where the estimator's own standard error is largest (per-spin payout SD
    runs 6-9 there, so 40k samples gives the calibrator an SE of ~0.03-0.045 —
    bigger than the whole effect being calibrated).

    The old docstring justified Monte Carlo on the grounds that a closed form
    "goes negative whenever a game combines a high bonus_frequency with a high
    max_multiplier". That was true of the relation it stated, which used
    0.65 * max_multiplier as the bonus term — but the sampler draws
    Beta(1,12), whose mean is 1/13 = 0.077, not 0.65. With the correct
    constant that formulation stays positive on all 15 games. And this one is
    unconditionally positive regardless, because payout_scale multiplies
    *both* branches, so the solve is target_rtp / E[raw] with E[raw] > 0
    whenever the game can pay out at all.

    _calibrate_payout_scale_monte_carlo is kept as the test oracle: an
    independent estimate that must agree with this to within its own
    sampling error.
    """
    raw = _expected_raw_return(win_prob, bonus_freq, max_mult)
    if raw <= 1e-9:
        return 1.0
    return target_rtp / raw


def _calibrate_payout_scale_monte_carlo(
    volatility: Volatility,
    win_prob: float,
    bonus_freq: float,
    max_mult: float,
    target_rtp: float,
    rng: np.random.Generator,
    n_samples: int = 40000,
) -> float:
    """Simulate n_samples spins at unit scale, measure the realised return per
    bet, then solve for the factor that brings it to target_rtp.

    No longer used to build the catalogue — see _payout_scale for why. Kept
    because an independent empirical estimate is exactly what you want to
    check an analytic derivation against, and tests/test_payout_calibration.py
    does that.
    """
    total = 0.0
    for _ in range(n_samples):
        if rng.random() < bonus_freq:
            total += rng.beta(_BONUS_BETA_A, _BONUS_BETA_B) * max_mult
        elif rng.random() < win_prob:
            total += _raw_shape_draw(volatility, rng)
    raw_return = total / n_samples
    if raw_return <= 1e-9:
        return 1.0
    return target_rtp / raw_return


def generate_games(n: int, rng: np.random.Generator) -> list[SlotGame]:
    """Build the game catalogue with payout math actually tied to RTP.

    Each game's RTP is the target long-run return per dollar bet, and
    payout_scale is solved exactly so that:

        rtp = payout_scale * [ bonus_freq * (max_multiplier / 13)
                             + (1 - bonus_freq) * win_prob ]

    (1/13 is E[Beta(1,12)], the bonus multiplier's mean as a fraction of
    max_multiplier; the non-bonus shape draws all have mean 1.0.)

    This makes RTP a real driver of the simulation rather than a decorative
    field, and it's what keeps high-volatility games "mostly losing, with a
    few outsized wins" rather than just randomly generous — the *shape* still
    differs per tier even though the long-run mean is pinned.
    """
    games = []
    tiers = [Volatility.LOW, Volatility.MEDIUM, Volatility.HIGH]
    for i in range(n):
        volatility = tiers[i % 3]
        theme = THEMES[i % len(THEMES)]
        if volatility is Volatility.LOW:
            bonus_freq = rng.uniform(0.025, 0.05)
            max_mult = rng.uniform(5, 20)
        elif volatility is Volatility.MEDIUM:
            bonus_freq = rng.uniform(0.015, 0.035)
            max_mult = rng.uniform(20, 100)
        else:
            bonus_freq = rng.uniform(0.005, 0.02)
            max_mult = rng.uniform(100, 500)

        rtp = round(rng.uniform(0.92, 0.97), 4)
        win_prob = round(rng.uniform(*_WIN_PROB_RANGE[volatility]), 4)
        bonus_freq = round(bonus_freq, 4)
        max_mult = round(max_mult, 1)

        payout_scale = _payout_scale(win_prob, bonus_freq, max_mult, rtp)

        games.append(SlotGame(
            game_id=f"game_{i:02d}",
            name=f"{theme.replace('_', ' ').title()} {volatility.value.title()}",
            theme=theme,
            volatility=volatility,
            rtp=rtp,
            base_win_probability=win_prob,
            bonus_frequency=bonus_freq,
            max_multiplier=max_mult,
            payout_scale=round(payout_scale, 5),
            min_bet=0.10,
            max_bet=round(rng.uniform(50, 200), 0),
        ))
    return games


def generate_players(
    personas: list[PlayerPersona], per_persona: int, rng: np.random.Generator
) -> list[Player]:
    players = []
    for persona in personas:
        for _ in range(per_persona):
            bankroll = rng.uniform(*persona.starting_bankroll_range)
            players.append(Player(
                player_id=f"player_{uuid.uuid4().hex[:8]}",
                persona_id=persona.persona_id,
                starting_bankroll=round(bankroll, 2),
            ))
    return players


def _pick_game(persona: PlayerPersona, games: list[SlotGame], rng: np.random.Generator) -> SlotGame:
    tiers = list(persona.volatility_preference.keys())
    weights = np.array([persona.volatility_preference[t] for t in tiers])
    weights = weights / weights.sum()
    # Sample an index rather than passing enum objects through np.choice —
    # numpy silently coerces str-Enum members to fixed-width numpy strings,
    # which breaks equality checks below.
    chosen_tier = tiers[rng.choice(len(tiers), p=weights)]
    candidates = [g for g in games if g.volatility == chosen_tier]
    return candidates[rng.integers(0, len(candidates))]


# The Beta quantile below assumes a == 1, where Beta(1, b) has the closed form
# CDF 1 - (1-x)^b. Guard it, because a future tweak to _BONUS_BETA_A would
# silently make spin_from_uniforms wrong rather than fail.
assert _BONUS_BETA_A == 1.0, "the closed-form Beta quantile below requires a == 1"


def _shape_quantile(volatility: Volatility, u: float) -> float:
    """Inverse CDF of the unscaled non-bonus shape, for u in [0, 1).

    The same three distributions as _raw_shape_draw, expressed as quantile
    functions instead of samplers. Feeding U(0,1) through here is exactly
    inverse-transform sampling, so the outputs are distributed identically —
    but the randomness is now an *input*, which is what makes a byte-for-byte
    cross-language test possible (see spin_from_uniforms).

        LOW     U(0.5, 1.5)   ->  0.5 + u
        MEDIUM  U(0.2, 1.8)   ->  0.2 + 1.6u
        HIGH    Exp(1)        ->  -ln(1 - u)

    log1p(-u) rather than log(1 - u): for small u the former keeps full
    precision where the latter loses it to cancellation.
    """
    if volatility is Volatility.LOW:
        return 0.5 + u
    if volatility is Volatility.MEDIUM:
        return 0.2 + 1.6 * u
    return -math.log1p(-u)


def spin_from_uniforms(
    game: SlotGame, u_bonus: float, u_win: float, u_shape: float
) -> tuple[bool, float, bool]:
    """The payout maths, as a pure function. Returns (is_win, multiplier, is_bonus).

    This is the single definition of how a spin pays out. _spin wraps it with
    an RNG, and the Rust port in rust/src/spin.rs is a line-by-line
    translation of it — so "do the two implementations agree?" becomes a
    question you can answer exactly, by feeding both the same three uniforms
    and comparing outputs, rather than only statistically by comparing
    histograms.

    Three uniforms are always consumed, even though at most one branch uses
    u_shape. That is distributionally identical to drawing lazily, since the
    three are independent and the branch tests never look at u_shape — and it
    keeps the signature fixed, which the golden-vector fixture depends on.

    Payout scale: both branches are multiplied by game.payout_scale, which
    _payout_scale solved exactly so that E[payout/bet] equals game.rtp. The
    *shape* still differs per volatility tier (tight for LOW, heavy-tailed for
    HIGH), so variance differs visibly even though the long-run mean is pinned.
    """
    if u_bonus < game.bonus_frequency:
        # Beta(1, b) quantile: 1 - (1-u)^(1/b).
        beta = 1.0 - (1.0 - u_shape) ** (1.0 / _BONUS_BETA_B)
        return True, beta * game.max_multiplier * game.payout_scale, True

    if u_win < game.base_win_probability:
        multiplier = _shape_quantile(game.volatility, u_shape) * game.payout_scale
        return True, max(multiplier, 0.0), False

    return False, 0.0, False


def _spin(game: SlotGame, rng: np.random.Generator) -> tuple[bool, float, bool]:
    """Draw three uniforms and hand them to the pure spin function.

    slot_machine.py imports this, so the live machine and the offline dataset
    keep sharing one definition of the payout maths.
    """
    return spin_from_uniforms(game, rng.random(), rng.random(), rng.random())


def simulate_session(
    player: Player,
    persona: PlayerPersona,
    game: SlotGame,
    rng: np.random.Generator,
) -> tuple[Session, list[BetEvent]]:
    session_id = f"session_{uuid.uuid4().hex[:10]}"
    start_bankroll = round(rng.uniform(*persona.starting_bankroll_range), 2)
    bankroll = start_bankroll

    base_bet = np.clip(
        start_bankroll * persona.base_bet_fraction_of_bankroll, game.min_bet, game.max_bet
    )
    max_bet_allowed = start_bankroll * persona.max_bet_fraction_of_bankroll
    bet = base_bet

    max_spins = int(rng.integers(*persona.session_length_spins_range))
    consecutive_losses = 0
    max_consecutive_losses = 0
    last_outcome: str | None = None
    bets_after_loss: list[float] = []
    bets_after_win: list[float] = []

    events: list[BetEvent] = []
    total_wagered = 0.0
    total_won = 0.0
    end_reason = EndReason.MAX_SPINS_REACHED

    for spin_number in range(1, max_spins + 1):
        if bankroll < bet:
            end_reason = EndReason.BANKROLL_DEPLETED
            break

        if last_outcome == "loss":
            bets_after_loss.append(bet)
        elif last_outcome == "win":
            bets_after_win.append(bet)

        is_win, multiplier, is_bonus = _spin(game, rng)
        payout = round(bet * multiplier, 2) if is_win else 0.0
        bankroll = round(bankroll - bet + payout, 2)
        total_wagered += bet
        total_won += payout

        events.append(BetEvent(
            event_id=f"{session_id}_{spin_number}",
            session_id=session_id,
            spin_number=spin_number,
            bet_amount=round(bet, 2),
            is_win=is_win,
            payout=payout,
            is_bonus_triggered=is_bonus,
            bankroll_after=bankroll,
        ))

        if is_win:
            consecutive_losses = 0
            last_outcome = "win"
            bet = bet * persona.bet_decrease_after_win
        else:
            consecutive_losses += 1
            max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
            last_outcome = "loss"
            bet = bet * persona.bet_increase_after_loss

        bet = float(np.clip(bet, game.min_bet, min(game.max_bet, max_bet_allowed)))

        if bankroll <= start_bankroll * persona.bankroll_stop_fraction:
            end_reason = EndReason.BANKROLL_DEPLETED
            break

        quit_prob = persona.base_quit_probability + (
            persona.loss_streak_quit_sensitivity * 0.01 * consecutive_losses
        )
        quit_prob = float(np.clip(quit_prob, 0.0, 0.9))
        if rng.random() < quit_prob:
            end_reason = EndReason.VOLUNTARY_QUIT
            break

    all_bets = [e.bet_amount for e in events]
    avg_bet = float(np.mean(all_bets)) if all_bets else 0.0
    bet_std = float(np.std(all_bets)) if all_bets else 0.0
    if bets_after_win:
        escalation_ratio = (
            float(np.mean(bets_after_loss)) / float(np.mean(bets_after_win))
            if bets_after_loss else 1.0
        )
    else:
        escalation_ratio = float("nan")

    session = Session(
        session_id=session_id,
        player_id=player.player_id,
        persona_id=persona.persona_id,
        game_id=game.game_id,
        start_bankroll=start_bankroll,
        end_bankroll=bankroll,
        num_spins=len(events),
        total_wagered=round(total_wagered, 2),
        total_won=round(total_won, 2),
        net_result=round(total_won - total_wagered, 2),
        avg_bet=round(avg_bet, 2),
        bet_std=round(bet_std, 2),
        escalation_ratio=round(escalation_ratio, 3) if escalation_ratio == escalation_ratio else None,
        max_consecutive_losses=max_consecutive_losses,
        end_reason=end_reason,
    )
    return session, events


def run(
    n_games: int,
    players_per_persona: int,
    sessions_per_player: int,
    seed: int,
    out_dir: Path,
) -> None:
    rng = np.random.default_rng(seed)
    personas_by_id = {p.persona_id: p for p in DEFAULT_PERSONAS}

    games = generate_games(n_games, rng)
    players = generate_players(DEFAULT_PERSONAS, players_per_persona, rng)

    all_sessions: list[Session] = []
    all_events: list[BetEvent] = []

    for player in players:
        persona = personas_by_id[player.persona_id]
        for _ in range(sessions_per_player):
            game = _pick_game(persona, games, rng)
            session, events = simulate_session(player, persona, game, rng)
            all_sessions.append(session)
            all_events.extend(events)

    out_dir.mkdir(parents=True, exist_ok=True)

    games_df = pd.DataFrame([vars(g) for g in games])
    games_df["volatility"] = games_df["volatility"].apply(lambda v: v.value if hasattr(v, "value") else v)

    players_df = pd.DataFrame([vars(p) for p in players])

    sessions_df = pd.DataFrame([vars(s) for s in all_sessions])
    sessions_df["end_reason"] = sessions_df["end_reason"].apply(
        lambda v: v.value if hasattr(v, "value") else v
    )

    events_df = pd.DataFrame([vars(e) for e in all_events])

    games_df.to_csv(out_dir / "games.csv", index=False)
    players_df.to_csv(out_dir / "players.csv", index=False)
    sessions_df.to_csv(out_dir / "sessions.csv", index=False)
    events_df.to_csv(out_dir / "events.csv", index=False)

    print(f"games:    {len(games_df):>6} rows -> {out_dir/'games.csv'}")
    print(f"players:  {len(players_df):>6} rows -> {out_dir/'players.csv'}")
    print(f"sessions: {len(sessions_df):>6} rows -> {out_dir/'sessions.csv'}")
    print(f"events:   {len(events_df):>6} rows -> {out_dir/'events.csv'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-games", type=int, default=15)
    parser.add_argument("--players-per-persona", type=int, default=30)
    parser.add_argument("--sessions-per-player", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=Path, default=Path("data"))
    args = parser.parse_args()

    run(
        n_games=args.n_games,
        players_per_persona=args.players_per_persona,
        sessions_per_player=args.sessions_per_player,
        seed=args.seed,
        out_dir=args.out_dir,
    )


if __name__ == "__main__":
    main()
