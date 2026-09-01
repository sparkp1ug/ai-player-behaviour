"""
Data schema for the synthetic player-behaviour dataset.

This module defines the shape of every entity in the simulation. Nothing in
here generates data — it's the contract that generate_synthetic_data.py
fills in. Keeping the schema separate makes it easy to reference from the
feature-engineering / clustering / recommendation stages later in the
pipeline without re-reading generator internals.
"""

from dataclasses import dataclass, field
from enum import Enum


class Volatility(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class EndReason(str, Enum):
    BANKROLL_DEPLETED = "bankroll_depleted"
    VOLUNTARY_QUIT = "voluntary_quit"
    MAX_SPINS_REACHED = "max_spins_reached"


# ---------------------------------------------------------------------------
# Game catalogue
# ---------------------------------------------------------------------------

@dataclass
class SlotGame:
    game_id: str
    name: str
    theme: str                 # e.g. "egyptian", "fruits", "mythology"
    volatility: Volatility
    rtp: float                 # theoretical return-to-player, e.g. 0.94
    base_win_probability: float  # per-spin chance of a non-bonus win; fixed
                                   # per game so RTP-calibration is stable
    bonus_frequency: float      # probability a spin triggers a bonus feature
    max_multiplier: float       # largest possible payout multiple of bet
    payout_scale: float          # Monte Carlo-calibrated scale factor so
                                   # E[payout/bet] tracks this game's rtp
                                   # (see generate_synthetic_data.py)
    min_bet: float
    max_bet: float


# ---------------------------------------------------------------------------
# Player personas — the "ground truth" behavioural archetypes.
# The clustering step later should roughly recover these from behaviour
# alone, without ever seeing persona_id directly.
# ---------------------------------------------------------------------------

@dataclass
class PlayerPersona:
    persona_id: str
    name: str
    description: str

    # Bankroll & bet sizing
    starting_bankroll_range: tuple[float, float]
    base_bet_fraction_of_bankroll: float   # initial bet as % of bankroll

    # Bet-adaptation behaviour (the core "tell" for behavioural clustering)
    bet_increase_after_loss: float   # multiplier applied to bet after a loss
    bet_decrease_after_win: float    # multiplier applied to bet after a win
    max_bet_fraction_of_bankroll: float  # ceiling, prevents runaway escalation

    # Game preference — sampling weights over volatility tiers
    volatility_preference: dict[Volatility, float]

    # Session behaviour
    session_length_spins_range: tuple[int, int]
    base_quit_probability: float          # per-spin chance of voluntarily stopping
    loss_streak_quit_sensitivity: float   # how much a loss streak changes quit prob
                                            # (positive = quits sooner when losing,
                                            #  negative = "chasing" — plays through it)
    bankroll_stop_fraction: float          # stop session if bankroll falls below
                                            # this fraction of the starting amount


# ---------------------------------------------------------------------------
# Simulated entities
# ---------------------------------------------------------------------------

@dataclass
class Player:
    player_id: str
    persona_id: str
    starting_bankroll: float


@dataclass
class BetEvent:
    event_id: str
    session_id: str
    spin_number: int
    bet_amount: float
    is_win: bool
    payout: float                # 0 if loss
    is_bonus_triggered: bool
    bankroll_after: float


@dataclass
class Session:
    session_id: str
    player_id: str
    persona_id: str
    game_id: str
    start_bankroll: float
    end_bankroll: float
    num_spins: int
    total_wagered: float
    total_won: float
    net_result: float                  # total_won - total_wagered
    avg_bet: float
    bet_std: float
    escalation_ratio: float            # avg bet-after-loss / avg bet-after-win
    max_consecutive_losses: int
    end_reason: EndReason


# ---------------------------------------------------------------------------
# Default persona set
# ---------------------------------------------------------------------------

DEFAULT_PERSONAS: list[PlayerPersona] = [
    PlayerPersona(
        persona_id="cautious_grinder",
        name="Cautious Grinder",
        description="Small, steady bets. Backs off quickly after losses. "
                    "Plays long sessions on low-volatility games.",
        starting_bankroll_range=(100, 300),
        base_bet_fraction_of_bankroll=0.01,
        bet_increase_after_loss=1.0,
        bet_decrease_after_win=0.95,
        max_bet_fraction_of_bankroll=0.02,
        volatility_preference={Volatility.LOW: 0.7, Volatility.MEDIUM: 0.25, Volatility.HIGH: 0.05},
        session_length_spins_range=(80, 250),
        base_quit_probability=0.01,
        loss_streak_quit_sensitivity=0.6,
        bankroll_stop_fraction=0.4,
    ),
    PlayerPersona(
        persona_id="volatility_chaser",
        name="Volatility Chaser",
        description="Seeks big multipliers on high-volatility games. "
                    "Bets grow after wins, chasing bigger hits.",
        starting_bankroll_range=(150, 500),
        base_bet_fraction_of_bankroll=0.02,
        bet_increase_after_loss=1.05,
        bet_decrease_after_win=1.15,
        max_bet_fraction_of_bankroll=0.12,
        volatility_preference={Volatility.LOW: 0.05, Volatility.MEDIUM: 0.25, Volatility.HIGH: 0.7},
        session_length_spins_range=(30, 120),
        base_quit_probability=0.015,
        loss_streak_quit_sensitivity=0.2,
        bankroll_stop_fraction=0.15,
    ),
    PlayerPersona(
        persona_id="casual_short_session",
        name="Casual / Short Session",
        description="Plays briefly, flat bets, low engagement with escalation.",
        starting_bankroll_range=(50, 150),
        base_bet_fraction_of_bankroll=0.015,
        bet_increase_after_loss=1.0,
        bet_decrease_after_win=1.0,
        max_bet_fraction_of_bankroll=0.02,
        volatility_preference={Volatility.LOW: 0.4, Volatility.MEDIUM: 0.45, Volatility.HIGH: 0.15},
        session_length_spins_range=(10, 40),
        base_quit_probability=0.05,
        loss_streak_quit_sensitivity=0.3,
        bankroll_stop_fraction=0.5,
    ),
    PlayerPersona(
        persona_id="bonus_hunter",
        name="Bonus Hunter",
        description="Chooses games with high bonus frequency, moderate bets, "
                    "switches games often looking for bonus rounds.",
        starting_bankroll_range=(100, 350),
        base_bet_fraction_of_bankroll=0.015,
        bet_increase_after_loss=1.02,
        bet_decrease_after_win=1.0,
        max_bet_fraction_of_bankroll=0.05,
        volatility_preference={Volatility.LOW: 0.2, Volatility.MEDIUM: 0.5, Volatility.HIGH: 0.3},
        session_length_spins_range=(40, 150),
        base_quit_probability=0.02,
        loss_streak_quit_sensitivity=0.25,
        bankroll_stop_fraction=0.3,
    ),
    PlayerPersona(
        persona_id="loss_chaser",
        name="Loss Chaser (at-risk pattern)",
        description="Escalates bets sharply after losses trying to win back "
                    "losses; plays through losing streaks rather than "
                    "stopping. Included so the responsible-play detector "
                    "has a genuine positive case to catch — not a persona "
                    "to optimise engagement for.",
        starting_bankroll_range=(100, 400),
        base_bet_fraction_of_bankroll=0.015,
        bet_increase_after_loss=1.35,
        bet_decrease_after_win=0.9,
        max_bet_fraction_of_bankroll=0.25,
        volatility_preference={Volatility.LOW: 0.1, Volatility.MEDIUM: 0.3, Volatility.HIGH: 0.6},
        session_length_spins_range=(50, 200),
        base_quit_probability=0.005,
        loss_streak_quit_sensitivity=-0.5,  # negative: keeps playing through losses
        bankroll_stop_fraction=0.05,
    ),
]
