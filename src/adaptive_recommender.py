"""
The recommender that learns while the game is being played.

recommendation_engine.py is a static, content-based matcher: build a vector
for the player, build a vector for each game, rank by cosine similarity. It's
a reasonable cold-start rule and it never improves, because nothing ever tells
it whether a recommendation was any good.

This module keeps that matcher and puts a contextual bandit around it:

    score(game) = prior_weight * cosine_prior(player, game)      cold start
                + shared LinUCB estimate from the player's context   learned
                + per-player residual correction                     personal
                + exploration bonus

* Shared term — LinUCB (Li et al., 2010). One ridge regression per game over
  the player's observable behaviour vector, with an optimism bonus
  alpha * sqrt(xᵀ A⁻¹ x) so games we know little about still get tried. This
  is what generalises: what it learns from one cautious grinder transfers to
  every other player who behaves like one.
* Personal term — a shrunk mean of this player's own reward residuals on this
  game, weight n/(n+k). Nothing at first, dominant once a player has real
  history on a game. (The "hybrid" idea from the same LinUCB paper, kept
  deliberately crude.)

  Theme play-history is deliberately *not* in the context vector. It looked
  like a clear win over three seeds and evaporated over six — the gain was
  sampling noise. The per-player residual already carries individual taste,
  and a 15-dim one-hot of past choices mostly re-describes what the
  recommender itself did last round.
* Safety layer — NOT learned, and not negotiable. Players the responsible-play
  model has flagged get high-volatility and high-max-multiplier games pushed
  down the ranking using the same constant as recommendation_engine.py. A
  bandit optimises the reward it is handed; anything that must hold regardless
  of reward belongs in a hard constraint, not in the objective.

The bandit updates on realised session satisfaction, so it tracks a moving
population: when tastes shift mid-run, the ridge estimates get pulled back
towards the new truth instead of staying frozen at the old one.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from recommendation_engine import HIGH_VOLATILITY_DOWNWEIGHT, cosine_sim  # noqa: E402

# Order matters: this is the context vector layout, shared by every arm.
CONTEXT_FEATURES = [
    "pref_low_volatility",
    "pref_medium_volatility",
    "pref_high_volatility",
    "avg_bet_fraction_of_bankroll",
    "avg_session_length",
    "avg_escalation_ratio",
    "max_consecutive_losses_mean",
    "pct_bankroll_depleted",
    "bonus_trigger_rate",
    "recent_satisfaction",
]

PRIOR_WEIGHT_START = 0.6     # how much the cosine prior counts at round 0
PRIOR_DECAY_PULLS = 40       # pulls after which the prior is ~halved for an arm
PERSONAL_SHRINKAGE = 1.0     # k in n/(n+k): a player's own history on a game
                             # outweighs the shared estimate from the 2nd play.
                             # Swept over 6 seeds against k=3; k=1 was worth
                             # about +0.003 satisfaction/session.
HIGH_MAX_MULT_PERCENTILE = 0.66   # matches recommendation_engine.py


def context_vector(context: dict[str, float]) -> np.ndarray:
    """dict -> fixed-layout array with a bias term appended."""
    x = np.array([float(context.get(f, 0.0)) for f in CONTEXT_FEATURES], dtype=float)
    return np.append(x, 1.0)


@dataclass
class _Arm:
    """Ridge regression state for one game."""
    A: np.ndarray            # d x d design matrix (X'X + I)
    b: np.ndarray            # d   response vector (X'y)
    pulls: int = 0
    personal: dict[str, tuple[int, float]] = field(default_factory=dict)  # player -> (n, mean residual)

    def theta(self) -> np.ndarray:
        return np.linalg.solve(self.A, self.b)


class AdaptiveRecommender:
    """Hybrid LinUCB over the game catalogue, warm-started from cosine similarity."""

    def __init__(
        self,
        game_ids: list[str],
        game_vectors: pd.DataFrame,
        alpha: float = 0.35,
        ridge: float = 1.0,
        prior_weight: float = PRIOR_WEIGHT_START,
        seed: int | None = None,
    ) -> None:
        self.game_ids = list(game_ids)
        self.alpha = alpha
        self.prior_weight = prior_weight
        self.rng = np.random.default_rng(seed)

        d = len(CONTEXT_FEATURES) + 1
        self.arms = {g: _Arm(A=ridge * np.eye(d), b=np.zeros(d)) for g in self.game_ids}

        # Content vectors from the existing static engine, used for the prior
        # and for the risk-aware safety layer.
        self._game_vecs = game_vectors.set_index("game_id").reindex(self.game_ids)
        self._prior_cols = [c for c in game_vectors.columns
                            if c not in ("game_id", "name", "vol_low", "vol_medium", "vol_high")]
        self._game_matrix = self._game_vecs[self._prior_cols].to_numpy(dtype=float)
        self._risk_appetite = self._game_vecs["risk_appetite"].to_numpy(dtype=float)
        self._high_mult_cut = float(np.quantile(self._risk_appetite, HIGH_MAX_MULT_PERCENTILE))
        self._is_high_vol = self._game_vecs["vol_high"].to_numpy(dtype=float) > 0
        self._volatility = np.select(
            [self._game_vecs["vol_low"].to_numpy(dtype=float) > 0,
             self._game_vecs["vol_medium"].to_numpy(dtype=float) > 0],
            ["low", "medium"], default="high",
        )

    # -- scoring -------------------------------------------------------------

    def _cosine_prior(self, context: dict[str, float]) -> np.ndarray:
        """Reuse the static engine's representation: a player vector in the
        same space as the game vectors (volatility/theme prefs are not all
        observable live, so this uses the parts that are)."""
        player = np.zeros(len(self._prior_cols))
        for i, col in enumerate(self._prior_cols):
            if col == "bonus_affinity":
                player[i] = context.get("bonus_trigger_rate", 0.0)
            elif col == "risk_appetite":
                player[i] = context.get("avg_bet_fraction_of_bankroll", 0.0)
            elif col.startswith("theme_"):
                player[i] = context.get(f"pref_{col}", 0.0)
        if not player.any():
            return np.zeros(len(self.game_ids))
        return cosine_sim(player.reshape(1, -1), self._game_matrix).flatten()

    def scores(self, context: dict[str, float], player_id: str, risk_flag: bool = False) -> np.ndarray:
        x = context_vector(context)
        prior = self._cosine_prior(context)
        out = np.empty(len(self.game_ids))

        for i, game_id in enumerate(self.game_ids):
            arm = self.arms[game_id]
            A_inv_x = np.linalg.solve(arm.A, x)
            mean = float(arm.theta() @ x)
            bonus = self.alpha * float(np.sqrt(max(x @ A_inv_x, 0.0)))

            # The cosine prior carries the arm early and fades as evidence
            # accumulates — a cold-start crutch, not a permanent thumb on the
            # scale.
            w_prior = self.prior_weight * PRIOR_DECAY_PULLS / (PRIOR_DECAY_PULLS + arm.pulls)

            n, resid = arm.personal.get(player_id, (0, 0.0))
            personal = resid * n / (n + PERSONAL_SHRINKAGE)

            out[i] = w_prior * prior[i] + mean + personal + bonus

        if risk_flag:
            out = self._apply_safety_layer(out)
        return out

    def _apply_safety_layer(self, scores: np.ndarray) -> np.ndarray:
        """Same rule as recommendation_engine.recommend_for_players: flagged
        players get high-volatility games downweighted, and the biggest
        max-multiplier games downweighted again on top."""
        adjusted = scores.copy()
        adjusted[self._is_high_vol] *= HIGH_VOLATILITY_DOWNWEIGHT
        adjusted[self._risk_appetite > self._high_mult_cut] *= HIGH_VOLATILITY_DOWNWEIGHT
        return adjusted

    def recommend(self, context: dict[str, float], player_id: str, risk_flag: bool = False) -> str:
        scores = self.scores(context, player_id, risk_flag)
        best = np.flatnonzero(scores == scores.max())
        return self.game_ids[int(best[self.rng.integers(0, len(best))])]

    def top_n(self, context: dict[str, float], player_id: str, risk_flag: bool = False,
              n: int = 5) -> pd.DataFrame:
        scores = self.scores(context, player_id, risk_flag)
        order = np.argsort(scores)[::-1][:n]
        return pd.DataFrame({
            "game_id": [self.game_ids[i] for i in order],
            "name": self._game_vecs["name"].to_numpy()[order],
            "volatility": self._volatility[order],
            "score": np.round(scores[order], 4),
            "pulls": [self.arms[self.game_ids[i]].pulls for i in order],
        })

    # -- learning ------------------------------------------------------------

    def update(self, context: dict[str, float], player_id: str, game_id: str, reward: float) -> None:
        x = context_vector(context)
        arm = self.arms[game_id]
        arm.A += np.outer(x, x)
        arm.b += reward * x
        arm.pulls += 1

        # Personal residual: what this player gets on this game over and above
        # what the shared model predicts for someone who behaves like them.
        residual = reward - float(arm.theta() @ x)
        n, mean = arm.personal.get(player_id, (0, 0.0))
        arm.personal[player_id] = (n + 1, mean + (residual - mean) / (n + 1))


# ---------------------------------------------------------------------------
# Baselines — the comparison that makes "it learns" a measurement, not a claim
# ---------------------------------------------------------------------------

class RandomRecommender:
    """Uniform over the catalogue. The floor."""

    def __init__(self, game_ids: list[str], seed: int | None = None) -> None:
        self.game_ids = list(game_ids)
        self.rng = np.random.default_rng(seed)

    def recommend(self, context: dict[str, float], player_id: str, risk_flag: bool = False) -> str:
        return self.game_ids[int(self.rng.integers(0, len(self.game_ids)))]

    def update(self, *args, **kwargs) -> None:  # noqa: D401 - deliberately inert
        pass


class StaticCosineRecommender:
    """The existing content-based engine, frozen. Same cold-start knowledge as
    the bandit's prior, but it never updates — so the gap between this and the
    bandit is exactly the value of learning from feedback."""

    def __init__(self, game_ids: list[str], game_vectors: pd.DataFrame, seed: int | None = None) -> None:
        self._inner = AdaptiveRecommender(game_ids, game_vectors, alpha=0.0,
                                          prior_weight=1.0, seed=seed)
        self.game_ids = list(game_ids)
        self.rng = self._inner.rng

    def recommend(self, context: dict[str, float], player_id: str, risk_flag: bool = False) -> str:
        return self._inner.recommend(context, player_id, risk_flag)

    def update(self, *args, **kwargs) -> None:
        pass
