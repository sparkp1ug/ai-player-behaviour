"""
A small, actually-playable slot machine.

Everything up to now in this repo has been *offline*: generate_synthetic_data.py
rolls a whole dataset in one shot, and the recommender reads the CSVs it left
behind. This module turns the same maths into a live environment you can spin
one lever-pull at a time — by hand from the terminal, or (the real point)
under a population of learning agents in play_loop.py.

The payout maths is deliberately NOT reimplemented here. It's imported from
generate_synthetic_data, so a spin served by this machine is drawn from exactly
the same RTP-calibrated distribution as a spin in the offline dataset. If the
two ever drifted apart, every model trained on the offline data would be
quietly mis-specified the moment it met the live machine.

Usage:
    python src/slot_machine.py --play                 # interactive, human
    python src/slot_machine.py --audit --spins 200000 # realised vs target RTP
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# The rest of src/ uses flat imports (`from schema import ...`), which only
# resolve when src/ is on the path. Put it there so this module works both as
# `python src/slot_machine.py` and as `from src.slot_machine import ...`.
_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from schema import SlotGame, Volatility  # noqa: E402
from generate_synthetic_data import _spin as _draw_outcome  # noqa: E402

# Purely cosmetic. The reels are rendered *from* the outcome, they don't
# determine it — the RNG decides win/bonus/multiplier first (see _draw_outcome)
# and the symbols are then chosen to be consistent with that result. Real slot
# machines work the same way round; the reels are a presentation layer.
REEL_SYMBOLS = ["🍒", "🔔", "⭐", "💎", "7️⃣"]
BONUS_SYMBOL = "🎁"


@dataclass(frozen=True)
class SpinResult:
    game_id: str
    bet: float
    is_win: bool
    is_bonus: bool
    multiplier: float          # payout as a multiple of the bet (0 on a loss)
    payout: float              # bet * multiplier, rounded to cents
    reels: tuple[str, str, str]

    @property
    def net(self) -> float:
        return round(self.payout - self.bet, 2)


def games_from_frame(games: pd.DataFrame) -> list[SlotGame]:
    """Rehydrate games.csv rows into the SlotGame dataclass the maths expects."""
    return [
        SlotGame(
            game_id=row.game_id,
            name=row.name_,
            theme=row.theme,
            volatility=Volatility(row.volatility),
            rtp=float(row.rtp),
            base_win_probability=float(row.base_win_probability),
            bonus_frequency=float(row.bonus_frequency),
            max_multiplier=float(row.max_multiplier),
            payout_scale=float(row.payout_scale),
            min_bet=float(row.min_bet),
            max_bet=float(row.max_bet),
        )
        # `name` collides with the itertuples index attribute, hence the rename.
        for row in games.rename(columns={"name": "name_"}).itertuples(index=False)
    ]


class SlotMachine:
    """The live environment. Holds the catalogue and the RNG; does NOT hold a
    bankroll — whoever is playing owns their own money and decides bet sizes.
    That split is what lets one machine serve a whole population of agents."""

    def __init__(self, games: list[SlotGame], seed: int | None = None) -> None:
        if not games:
            raise ValueError("SlotMachine needs at least one game")
        self.games: dict[str, SlotGame] = {g.game_id: g for g in games}
        self.rng = np.random.default_rng(seed)
        self.total_spins = 0
        self.total_wagered = 0.0
        self.total_paid = 0.0

    @classmethod
    def from_csv(cls, games_csv: Path, seed: int | None = None) -> "SlotMachine":
        return cls(games_from_frame(pd.read_csv(games_csv)), seed=seed)

    @property
    def game_ids(self) -> list[str]:
        return list(self.games)

    @property
    def realised_rtp(self) -> float:
        """Paid out per unit wagered so far. Converges to the catalogue RTP
        given enough spins — that convergence is what --audit checks."""
        return self.total_paid / self.total_wagered if self.total_wagered else 0.0

    def spin(self, game_id: str, bet: float) -> SpinResult:
        game = self.games[game_id]
        bet = round(float(np.clip(bet, game.min_bet, game.max_bet)), 2)

        is_win, multiplier, is_bonus = _draw_outcome(game, self.rng)
        payout = round(bet * multiplier, 2) if is_win else 0.0

        self.total_spins += 1
        self.total_wagered += bet
        self.total_paid += payout

        return SpinResult(
            game_id=game_id,
            bet=bet,
            is_win=is_win,
            is_bonus=is_bonus,
            multiplier=round(multiplier, 4) if is_win else 0.0,
            payout=payout,
            reels=self._render_reels(is_win, is_bonus),
        )

    def _render_reels(self, is_win: bool, is_bonus: bool) -> tuple[str, str, str]:
        if is_bonus:
            return (BONUS_SYMBOL,) * 3
        if is_win:
            sym = REEL_SYMBOLS[self.rng.integers(0, len(REEL_SYMBOLS))]
            return (sym, sym, sym)
        # A losing spin: draw three symbols that are not all equal.
        while True:
            reels = tuple(
                REEL_SYMBOLS[i] for i in self.rng.integers(0, len(REEL_SYMBOLS), size=3)
            )
            if len(set(reels)) > 1:
                return reels  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Terminal front-ends
# ---------------------------------------------------------------------------

def _print_catalogue(machine: SlotMachine) -> None:
    print(f"{'id':<9}{'name':<24}{'vol':<8}{'rtp':<8}{'bonus':<8}{'max x'}")
    for game in machine.games.values():
        print(f"{game.game_id:<9}{game.name:<24}{game.volatility.value:<8}"
              f"{game.rtp:<8.4f}{game.bonus_frequency:<8.4f}{game.max_multiplier:.0f}x")


def play_interactive(machine: SlotMachine, game_id: str, bankroll: float, bet: float) -> None:
    """Human mode. Enter to spin, 'b <amount>' to change bet, 'q' to walk away."""
    game = machine.games[game_id]
    start = bankroll
    print(f"\n{game.name}  ({game.volatility.value} volatility, "
          f"RTP {game.rtp:.2%}, bonus {game.bonus_frequency:.2%}, up to {game.max_multiplier:.0f}x)")
    print(f"bankroll {bankroll:.2f}   bet {bet:.2f}")
    print("[enter] spin   [b <amount>] change bet   [q] quit\n")

    spins = 0
    while bankroll >= bet:
        cmd = input("> ").strip().lower()
        if cmd == "q":
            break
        if cmd.startswith("b"):
            try:
                bet = round(float(np.clip(float(cmd[1:]), game.min_bet, game.max_bet)), 2)
                print(f"bet is now {bet:.2f}")
            except ValueError:
                print("usage: b 2.50")
            continue

        result = machine.spin(game_id, bet)
        bankroll = round(bankroll - result.bet + result.payout, 2)
        spins += 1

        reels = " ".join(result.reels)
        if result.is_bonus:
            tag = f"BONUS!  +{result.payout:.2f} ({result.multiplier:.1f}x)"
        elif result.is_win:
            tag = f"win     +{result.payout:.2f} ({result.multiplier:.2f}x)"
        else:
            tag = f"        -{result.bet:.2f}"
        print(f"  {reels}   {tag:<32} bankroll {bankroll:.2f}")

    if bankroll < bet:
        print("\nout of money.")
    print(f"\n{spins} spins, bankroll {start:.2f} -> {bankroll:.2f} "
          f"(net {bankroll - start:+.2f}), machine RTP so far {machine.realised_rtp:.2%}")


def audit(machine: SlotMachine, spins: int, bet: float = 1.0) -> pd.DataFrame:
    """Spin every game `spins` times at a flat bet and compare the realised
    return against the catalogue RTP. This is the honest check that the live
    machine pays out the way the game catalogue claims it does."""
    rows = []
    for game_id, game in machine.games.items():
        wagered = paid = 0.0
        wins = bonuses = 0
        biggest = 0.0
        for _ in range(spins):
            r = machine.spin(game_id, bet)
            wagered += r.bet
            paid += r.payout
            wins += r.is_win
            bonuses += r.is_bonus
            biggest = max(biggest, r.multiplier)
        rows.append({
            "game_id": game_id,
            "volatility": game.volatility.value,
            "target_rtp": game.rtp,
            "realised_rtp": round(paid / wagered, 4),
            "error": round(paid / wagered - game.rtp, 4),
            "hit_rate": round(wins / spins, 4),
            "bonus_rate": round(bonuses / spins, 4),
            "biggest_win_x": round(biggest, 1),
        })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--games-csv", type=Path, default=Path("data/games.csv"))
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--play", action="store_true", help="interactive human mode")
    parser.add_argument("--audit", action="store_true", help="realised vs target RTP per game")
    parser.add_argument("--list", action="store_true", help="print the game catalogue")
    parser.add_argument("--game", default="game_00", help="game_id for --play")
    parser.add_argument("--bankroll", type=float, default=200.0)
    parser.add_argument("--bet", type=float, default=2.0)
    parser.add_argument("--spins", type=int, default=100_000, help="spins per game for --audit")
    args = parser.parse_args()

    machine = SlotMachine.from_csv(args.games_csv, seed=args.seed)

    if args.list or not (args.play or args.audit):
        _print_catalogue(machine)
    if args.audit:
        table = audit(machine, args.spins)
        print()
        print(table.to_string(index=False))
        print(f"\nmean |realised - target| RTP error over {args.spins:,} spins/game: "
              f"{table['error'].abs().mean():.4f}")
    if args.play:
        play_interactive(machine, args.game, args.bankroll, args.bet)


if __name__ == "__main__":
    main()
