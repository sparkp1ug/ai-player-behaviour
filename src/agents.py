"""
The AI players.

Each agent wraps one of the behavioural personas from schema.py — that fixes
*how* it bets (escalation after a loss, when it walks away, how much of its
bankroll it risks) so the sessions it produces stay directly comparable with
the offline dataset. On top of that fixed temperament, three things about an
agent actually learn:

  1. Game taste (Q-values).  Each agent keeps a running value estimate per
     game, updated from how satisfying each session turned out. Unplayed games
     start optimistically, so a fresh agent explores before it settles.
  2. Trust in the recommender.  Agents don't blindly do as they're told. They
     accept a recommendation based on how it compares with their own current
     favourite, scaled by a trust level that rises when past recommendations
     landed well and falls when they didn't. Accept-rate is therefore an
     outcome the recommender has to earn, not a given.
  3. Boldness.  A multiplier on base bet size that drifts with recent results.
     Chasing personas get bolder after going bust; cautious ones pull back.
     This is what makes the population *non-stationary* — the recommender
     cannot learn a fixed answer once and coast.

Satisfaction (the reward signal) is deliberately not "money wagered" or "time
on device". Optimising an operator metric is exactly the failure mode this
repo is written to avoid. Reward here is session *quality* from the player's
own point of view — did the game suit their taste, did it last, was it fun —
with explicit penalties for sessions that show at-risk patterns. A recommender
trained on this signal is pushed away from harm rather than towards it.
"""

from __future__ import annotations

import sys
import uuid
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from schema import BetEvent, EndReason, PlayerPersona, Session, SlotGame, Volatility  # noqa: E402
from slot_machine import SlotMachine  # noqa: E402

# --- reward shaping ---------------------------------------------------------

SATISFACTION_WEIGHTS = {
    "volatility_fit": 0.30,   # does the game's risk profile match the persona
    "theme_fit": 0.20,        # the personal, unobservable part — what the
                              # recommender has to infer from behaviour alone
    "engagement": 0.20,       # played a full session rather than busting out
    "bonus_excitement": 0.20, # bonus rounds are the fun bit
    "outcome_feel": 0.10,     # winning feels better, but it's a minor term:
                              # a good recommendation can't be defined as a
                              # lucky one, or the model just learns noise
}

# Subtracted from satisfaction. These are the whole reason the reward is a
# player-welfare signal and not an engagement signal.
BUST_PENALTY = 0.15            # session ended with the bankroll gone
CHASING_PENALTY = 0.10         # bets after losses ran well above bets after wins
CHASING_ESCALATION_THRESHOLD = 1.25

# Optimistic initial value for a game an agent has never played: above the
# typical realised satisfaction, so untried games look worth a go.
OPTIMISTIC_INIT = 0.55
TASTE_LEARNING_RATE = 0.35
TRUST_LEARNING_RATE = 0.20
MIN_BANKROLL_TO_PLAY = 20.0

# Agents don't start knowing the whole catalogue — they know a handful of games
# and stumble onto new ones only rarely by themselves. This is the bit that
# makes a recommender worth having: without it, an agent left alone will have
# tried all 15 games within a few rounds and learned its own favourite, and no
# recommender can beat that. With it, discovery is the scarce resource and
# *which* unfamiliar game gets suggested is the whole game.
INITIAL_KNOWN_GAMES = 3
SELF_DISCOVERY_RATE = 0.05     # per-session chance of wandering off-catalogue alone

# Persona-level theme leanings. Themes are grouped so that a persona's taste is
# *statistically* predictable from its behaviour — a contextual model can learn
# "players who behave like this tend to enjoy these games" — while each agent
# still gets its own noisy draw around that base, so nothing is deterministic.
PERSONA_THEME_LEANINGS: dict[str, list[str]] = {
    "cautious_grinder": ["fruits", "animals", "holiday", "mining"],
    "volatility_chaser": ["aztec", "vikings", "space", "sports"],
    "casual_short_session": ["holiday", "animals", "fruits", "ocean"],
    "bonus_hunter": ["egyptian", "asian_treasures", "mythology", "pirates"],
    "loss_chaser": ["aztec", "egyptian", "western", "fantasy"],
}
LEANING_WEIGHT = 4.0      # favoured themes start this many times more likely
AFFINITY_NOISE = 0.45     # lognormal sigma on the per-agent draw


@dataclass
class SessionOutcome:
    session: Session
    events: list[BetEvent]
    satisfaction: float
    components: dict[str, float]


def satisfaction(
    persona: PlayerPersona,
    game: SlotGame,
    theme_affinity: dict[str, float],
    session: Session,
    events: list[BetEvent],
) -> tuple[float, dict[str, float]]:
    """Score one finished session from the player's point of view, in [0, 1]."""
    top_vol_pref = max(persona.volatility_preference.values()) or 1.0
    volatility_fit = persona.volatility_preference.get(game.volatility, 0.0) / top_vol_pref

    top_affinity = max(theme_affinity.values()) or 1.0
    theme_fit = theme_affinity.get(game.theme, 0.0) / top_affinity

    # A session that runs to its intended length scores 1.0; one cut short by
    # an empty bankroll scores proportionally less.
    intended = float(np.mean(persona.session_length_spins_range))
    engagement = float(np.clip(session.num_spins / intended, 0.0, 1.0))

    n_bonus = sum(e.is_bonus_triggered for e in events)
    bonus_excitement = float(1.0 - np.exp(-n_bonus / 1.5))  # saturates fast

    # Break-even reads as 0.5; losing everything wagered reads as 0.
    net_ratio = session.net_result / session.total_wagered if session.total_wagered else 0.0
    outcome_feel = float(np.clip(0.5 + net_ratio, 0.0, 1.0))

    components = {
        "volatility_fit": volatility_fit,
        "theme_fit": theme_fit,
        "engagement": engagement,
        "bonus_excitement": bonus_excitement,
        "outcome_feel": outcome_feel,
    }
    score = sum(SATISFACTION_WEIGHTS[k] * v for k, v in components.items())

    penalty = 0.0
    if session.end_reason == EndReason.BANKROLL_DEPLETED:
        penalty += BUST_PENALTY
    if session.escalation_ratio and session.escalation_ratio > CHASING_ESCALATION_THRESHOLD:
        penalty += CHASING_PENALTY
    components["penalty"] = penalty

    return float(np.clip(score - penalty, 0.0, 1.0)), components


class PlayerAgent:
    """One simulated player: a fixed persona plus the parts that adapt."""

    def __init__(
        self,
        agent_id: str,
        persona: PlayerPersona,
        theme_affinity: dict[str, float],
        starting_bankroll: float,
        rng: np.random.Generator,
        known_games: list[str],
        epsilon: float = 0.12,
    ) -> None:
        self.agent_id = agent_id
        self.persona = persona
        self.theme_affinity = theme_affinity
        self.rng = rng
        self.epsilon = epsilon

        self.bankroll = starting_bankroll
        self.starting_bankroll = starting_bankroll
        self.deposits = 1
        self.boldness = 1.0
        self.trust = 0.5                       # P(follow a neutral recommendation)

        self.known: set[str] = set(known_games)   # games this agent has heard of
        self.q: dict[str, float] = {}          # game_id -> value estimate
        self.plays: dict[str, int] = {}
        self.history: list[SessionOutcome] = []
        self.recent_satisfaction: float = 0.5
        self._game_volatility: dict[str, str] = {}   # filled by bind_catalogue
        self._game_theme: dict[str, str] = {}

        # Running behavioural stats. These mirror FEATURE_COLUMNS in
        # feature_engineering.py, computed online instead of batched over CSVs,
        # so the live context vector and the offline feature table mean the
        # same thing.
        self._vol_counts = {v: 0 for v in Volatility}
        self._theme_counts: dict[str, int] = {}
        self._n_sessions = 0
        self._sum_bet_fraction = 0.0
        self._sum_spins = 0
        self._sum_escalation = 0.0
        self._n_escalation = 0
        self._sum_max_loss_streak = 0
        self._n_busts = 0
        self._n_bonus = 0
        self._n_spins_total = 0

    # -- state the recommender is allowed to see -----------------------------

    def context(self) -> dict[str, float]:
        """Observable behaviour only. The persona label and the hidden theme
        affinity are NOT in here — inferring taste from behaviour is the whole
        job, and leaking the answer would make the experiment meaningless."""
        n = max(self._n_sessions, 1)
        # Theme play-history is observable (we know what they played) and the
        # cold-start cosine prior uses it. It is deliberately kept out of
        # CONTEXT_FEATURES, the bandit's own context: a 15-dim one-hot of past
        # choices mostly re-describes what the recommender already did, and
        # the per-player residual term covers individual taste more directly.
        themes = {f"pref_theme_{t}": c / n for t, c in self._theme_counts.items()}
        return themes | {
            "pref_low_volatility": self._vol_counts[Volatility.LOW] / n,
            "pref_medium_volatility": self._vol_counts[Volatility.MEDIUM] / n,
            "pref_high_volatility": self._vol_counts[Volatility.HIGH] / n,
            "avg_bet_fraction_of_bankroll": self._sum_bet_fraction / n,
            "avg_session_length": self._sum_spins / n / 100.0,     # ~unit scale
            "avg_escalation_ratio": (
                self._sum_escalation / self._n_escalation if self._n_escalation else 1.0
            ),
            "max_consecutive_losses_mean": self._sum_max_loss_streak / n / 10.0,
            "pct_bankroll_depleted": self._n_busts / n,
            "bonus_trigger_rate": (
                self._n_bonus / self._n_spins_total if self._n_spins_total else 0.0
            ),
            "recent_satisfaction": self.recent_satisfaction,
        }

    # -- choosing what to play ----------------------------------------------

    def value_of(self, game_id: str) -> float:
        return self.q.get(game_id, OPTIMISTIC_INIT)

    def own_choice(self, game_ids: list[str]) -> str:
        """What the agent would play unprompted: epsilon-greedy over the games
        it already knows, with a small chance of discovering one on its own."""
        unknown = [g for g in game_ids if g not in self.known]
        if unknown and self.rng.random() < SELF_DISCOVERY_RATE:
            return unknown[self.rng.integers(0, len(unknown))]

        known = sorted(self.known) or list(game_ids)
        if self.rng.random() < self.epsilon:
            return known[self.rng.integers(0, len(known))]
        values = np.array([self.value_of(g) for g in known])
        best = np.flatnonzero(values == values.max())
        return known[best[self.rng.integers(0, len(best))]]

    def consider(self, recommended: str | None, game_ids: list[str]) -> tuple[str, bool]:
        """Take the recommendation, or go own way. Returns (game_id, accepted)."""
        if recommended is None:
            return self.own_choice(game_ids), False

        own_best = max((self.value_of(g) for g in self.known), default=OPTIMISTIC_INIT)
        edge = self.value_of(recommended) - own_best        # usually <= 0
        # Trust shifts the intercept: a well-performing recommender gets the
        # benefit of the doubt on games the agent hasn't learned to like yet.
        p_accept = 1.0 / (1.0 + np.exp(-(6.0 * edge + 4.0 * (self.trust - 0.5))))
        if self.rng.random() < p_accept:
            return recommended, True
        return self.own_choice(game_ids), False

    # -- playing -------------------------------------------------------------

    def _ensure_funded(self) -> None:
        if self.bankroll < MIN_BANKROLL_TO_PLAY:
            self.bankroll = round(
                float(self.rng.uniform(*self.persona.starting_bankroll_range)), 2
            )
            self.deposits += 1

    def play(self, machine: SlotMachine, game_id: str) -> SessionOutcome:
        """Play one session on the live machine.

        Same bet dynamics as simulate_session() in generate_synthetic_data.py —
        escalate after a loss, decay after a win, quit probability that moves
        with the loss streak — with two differences that only matter for an
        agent that persists across rounds: the bankroll carries over between
        sessions (topped up when it runs dry), and base bet size is scaled by
        the agent's learned boldness.
        """
        self._ensure_funded()
        persona, game = self.persona, machine.games[game_id]

        session_id = f"session_{uuid.uuid4().hex[:10]}"
        start_bankroll = round(self.bankroll, 2)
        bankroll = start_bankroll

        base_bet = float(np.clip(
            start_bankroll * persona.base_bet_fraction_of_bankroll * self.boldness,
            game.min_bet, game.max_bet,
        ))
        # The persona's own ceiling stays fixed: boldness may change how big a
        # bet an agent opens with, never how far it is allowed to escalate.
        max_bet_allowed = start_bankroll * persona.max_bet_fraction_of_bankroll
        bet = base_bet

        max_spins = int(self.rng.integers(*persona.session_length_spins_range))
        consecutive_losses = 0
        max_consecutive_losses = 0
        last_outcome: str | None = None
        bets_after_loss: list[float] = []
        bets_after_win: list[float] = []

        events: list[BetEvent] = []
        total_wagered = total_won = 0.0
        end_reason = EndReason.MAX_SPINS_REACHED

        for spin_number in range(1, max_spins + 1):
            if bankroll < bet:
                end_reason = EndReason.BANKROLL_DEPLETED
                break

            if last_outcome == "loss":
                bets_after_loss.append(bet)
            elif last_outcome == "win":
                bets_after_win.append(bet)

            result = machine.spin(game_id, bet)
            bankroll = round(bankroll - result.bet + result.payout, 2)
            total_wagered += result.bet
            total_won += result.payout

            events.append(BetEvent(
                event_id=f"{session_id}_{spin_number}",
                session_id=session_id,
                spin_number=spin_number,
                bet_amount=result.bet,
                is_win=result.is_win,
                payout=result.payout,
                is_bonus_triggered=result.is_bonus,
                bankroll_after=bankroll,
            ))

            if result.is_win:
                consecutive_losses = 0
                last_outcome = "win"
                bet *= persona.bet_decrease_after_win
            else:
                consecutive_losses += 1
                max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
                last_outcome = "loss"
                bet *= persona.bet_increase_after_loss

            bet = float(np.clip(bet, game.min_bet, min(game.max_bet, max_bet_allowed)))

            if bankroll <= start_bankroll * persona.bankroll_stop_fraction:
                end_reason = EndReason.BANKROLL_DEPLETED
                break

            quit_prob = persona.base_quit_probability + (
                persona.loss_streak_quit_sensitivity * 0.01 * consecutive_losses
            )
            if self.rng.random() < float(np.clip(quit_prob, 0.0, 0.9)):
                end_reason = EndReason.VOLUNTARY_QUIT
                break

        bets = [e.bet_amount for e in events]
        if bets_after_win:
            escalation_ratio = (
                float(np.mean(bets_after_loss)) / float(np.mean(bets_after_win))
                if bets_after_loss else 1.0
            )
        else:
            escalation_ratio = float("nan")

        session = Session(
            session_id=session_id,
            player_id=self.agent_id,
            persona_id=persona.persona_id,
            game_id=game_id,
            start_bankroll=start_bankroll,
            end_bankroll=bankroll,
            num_spins=len(events),
            total_wagered=round(total_wagered, 2),
            total_won=round(total_won, 2),
            net_result=round(total_won - total_wagered, 2),
            avg_bet=round(float(np.mean(bets)), 2) if bets else 0.0,
            bet_std=round(float(np.std(bets)), 2) if bets else 0.0,
            escalation_ratio=round(escalation_ratio, 3) if escalation_ratio == escalation_ratio else None,
            max_consecutive_losses=max_consecutive_losses,
            end_reason=end_reason,
        )
        self.bankroll = bankroll

        score, components = satisfaction(persona, game, self.theme_affinity, session, events)
        return SessionOutcome(session=session, events=events, satisfaction=score, components=components)

    # -- adapting ------------------------------------------------------------

    def learn(self, game_id: str, outcome: SessionOutcome, accepted_recommendation: bool) -> None:
        s, session = outcome.satisfaction, outcome.session

        self.known.add(game_id)
        self.q[game_id] = self.value_of(game_id) + TASTE_LEARNING_RATE * (s - self.value_of(game_id))
        self.plays[game_id] = self.plays.get(game_id, 0) + 1
        self.recent_satisfaction += 0.3 * (s - self.recent_satisfaction)

        if accepted_recommendation:
            # Trust tracks whether following advice beat this agent's own
            # running expectation, not whether the session was good in absolute
            # terms — an agent on a cold streak shouldn't sour on the system.
            self.trust = float(np.clip(
                self.trust + TRUST_LEARNING_RATE * (s - self.recent_satisfaction), 0.05, 0.95
            ))

        # Boldness drift. Chasers (negative loss-streak quit sensitivity) bet
        # up after going bust; everyone else backs off. This is the mechanism
        # that keeps the population moving under the recommender's feet.
        is_chaser = self.persona.loss_streak_quit_sensitivity < 0
        if session.end_reason == EndReason.BANKROLL_DEPLETED:
            self.boldness *= 1.10 if is_chaser else 0.92
        elif session.net_result > 0:
            self.boldness *= 1.04 if self.persona.bet_decrease_after_win > 1.0 else 0.99
        self.boldness = float(np.clip(self.boldness, 0.5, 2.5))

        # Running behavioural stats for the next context vector.
        self._n_sessions += 1
        self._vol_counts[Volatility(self._volatility_of(game_id))] += 1
        theme = self._game_theme[game_id]
        self._theme_counts[theme] = self._theme_counts.get(theme, 0) + 1
        if session.start_bankroll:
            self._sum_bet_fraction += session.avg_bet / session.start_bankroll
        self._sum_spins += session.num_spins
        if session.escalation_ratio is not None:
            self._sum_escalation += session.escalation_ratio
            self._n_escalation += 1
        self._sum_max_loss_streak += session.max_consecutive_losses
        self._n_busts += session.end_reason == EndReason.BANKROLL_DEPLETED
        self._n_bonus += sum(e.is_bonus_triggered for e in outcome.events)
        self._n_spins_total += session.num_spins
        self.history.append(outcome)

    def _volatility_of(self, game_id: str) -> str:
        if not self._game_volatility:
            raise RuntimeError("call bind_catalogue(machine) before the agent plays")
        return self._game_volatility[game_id]

    def bind_catalogue(self, machine: SlotMachine) -> None:
        """Cache game_id -> volatility so stats can be updated without holding
        a reference to the machine."""
        self._game_volatility = {g.game_id: g.volatility.value for g in machine.games.values()}
        self._game_theme = {g.game_id: g.theme for g in machine.games.values()}

    def shift_tastes(self, themes: list[str], rng: np.random.Generator) -> None:
        """Rotate this agent's theme affinities onto different themes, keeping
        the same shape. Used by play_loop to stage a mid-run preference shock:
        anything the recommender learned about themes goes stale at once, and
        we get to watch whether it recovers."""
        shift = int(rng.integers(1, len(themes)))
        rotated = {themes[(i + shift) % len(themes)]: self.theme_affinity[t]
                   for i, t in enumerate(themes)}
        self.theme_affinity = rotated
        # Forget learned game values too — the agent's own taste changed.
        self.q = {g: OPTIMISTIC_INIT for g in self.q}


def build_agents(
    personas: list[PlayerPersona],
    per_persona: int,
    themes: list[str],
    game_ids: list[str],
    rng: np.random.Generator,
) -> list[PlayerAgent]:
    agents: list[PlayerAgent] = []
    for persona in personas:
        leaning = PERSONA_THEME_LEANINGS.get(persona.persona_id, [])
        base = np.array([LEANING_WEIGHT if t in leaning else 1.0 for t in themes])
        for _ in range(per_persona):
            weights = base * rng.lognormal(0.0, AFFINITY_NOISE, size=len(themes))
            weights = weights / weights.sum()
            known = [game_ids[i] for i in rng.choice(
                len(game_ids), size=min(INITIAL_KNOWN_GAMES, len(game_ids)), replace=False)]
            agents.append(PlayerAgent(
                agent_id=f"agent_{uuid.uuid4().hex[:8]}",
                persona=persona,
                theme_affinity=dict(zip(themes, weights)),
                starting_bankroll=round(float(rng.uniform(*persona.starting_bankroll_range)), 2),
                rng=np.random.default_rng(rng.integers(0, 2**32 - 1)),
                known_games=known,
            ))
    return agents
