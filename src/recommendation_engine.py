from __future__ import annotations
import pandas as pd

import numpy as np
import argparse
from pathlib import Path

VOLATILITY_TIERS = ["low", "medium", "high"]

HIGH_VOLATILITY_DOWNWEIGHT = 0.5   # multiply score by this for flagged players
HIGH_MAX_MULT_PERCENTILE = 0.66     # games above this max_multiplier percentile

def build_game_vectors(games: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    themes = sorted(games["theme"].unique())
    theme_cols = [f"theme_{t}" for t in themes]

    vol_onehot = pd.get_dummies(games["volatility"]).reindex(columns=VOLATILITY_TIERS, fill_value=0)
    vol_onehot.columns = [f"vol_{c}" for c in vol_onehot.columns]

    theme_onehot = pd.get_dummies(games["theme"]).reindex(columns=themes, fill_value=0)
    theme_onehot.columns = theme_cols

    bonus_affinity = games["bonus_frequency"] / games["bonus_frequency"].max()
    risk_appetite = games["max_multiplier"] / games["max_multiplier"].max()

    vec = pd.concat([vol_onehot, theme_onehot], axis=1)
    vec["bonus_affinity"] = bonus_affinity.values
    vec["risk_appetite"] = risk_appetite.values
    vec.insert(0, "game_id", games["game_id"].values)
    vec.insert(1, "name", games["name"].values)
    return vec, theme_cols

def build_player_vectors(
    sessions: pd.DataFrame,
    games: pd.DataFrame,
    player_features_raw: pd.DataFrame,
    theme_cols: list[str],
) -> pd.DataFrame:
    merged = sessions.merge(games[["game_id", "volatility", "theme"]], on="game_id", how="left")

    vol_pref = (
        merged.groupby("player_id")["volatility"]
        .value_counts(normalize=True)
        .unstack(fill_value=0.0)
        .reindex(columns=VOLATILITY_TIERS, fill_value=0.0)
    )
    vol_pref.columns = [f"vol_{c}" for c in vol_pref.columns]

    theme_pref = (
        merged.groupby("player_id")["theme"]
        .value_counts(normalize=True)
        .unstack(fill_value=0.0)
    )
    theme_pref.columns = [f"theme_{c}" for c in theme_pref.columns]
    theme_pref = theme_pref.reindex(columns=theme_cols, fill_value=0.0)

    risk_feats = player_features_raw.set_index("player_id")[["bonus_trigger_rate", "avg_bet_fraction_of_bankroll"]]
    max_bet_fraction = player_features_raw["avg_bet_fraction_of_bankroll"].max() or 1.0
    bonus_affinity = risk_feats["bonus_trigger_rate"].clip(0.0, 1.0)
    risk_appetite = (risk_feats["avg_bet_fraction_of_bankroll"] / max_bet_fraction).clip(0.0, 1.0)

    vec = vol_pref.join(theme_pref, how="outer").fillna(0.0)
    vec["bonus_affinity"] = bonus_affinity.reindex(vec.index).fillna(0.0)
    vec["risk_appetite"] = risk_appetite.reindex(vec.index).fillna(0.0)
    vec = vec.reset_index().rename(columns={"index": "player_id"})
    return vec

def cosine_sim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between two 2D arrays of vectors."""
    a_norm = a / np.linalg.norm(a, axis=1, keepdims=True)
    b_norm = b / np.linalg.norm(b, axis=1, keepdims=True)
    return np.dot(a_norm, b_norm.T)

def recommend_for_players(player_id: str, player_vecs: pd.DataFrame, game_vecs: pd.DataFrame, risk_table: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
    """Recommend games for a given player based on cosine similarity of feature vectors."""
    feature_cols = [col for col in game_vecs.columns if col not in ["game_id", "name", "vol_low", "vol_medium", "vol_high"]]
    player_row = player_vecs.set_index("player_id").loc[player_id]
    player_vector = player_row[feature_cols].values.astype(float)
    game_matrix = game_vecs[feature_cols].values.astype(float)

    scores = cosine_sim(player_vector.reshape(1, -1), game_matrix).flatten()
    result = game_vecs[["game_id", "name", "vol_low", "vol_medium", "vol_high"]].copy()
    result["score"] = scores

    is_flagged = False

    if risk_table is not None and player_id in risk_table["player_id"].values:
        is_flagged = risk_table.set_index("player_id").loc[player_id]["risk_flag"]

    if is_flagged:
        # Two compounding downweights, both applied to adjusted_score so the
        # raw similarity in `score` stays inspectable next to it:
        #   1. any high-volatility game
        #   2. any game whose top multiplier sits in the upper tail
        #
        # Both are written as in-place multiplications on a full-length array.
        # Assigning a *filtered* Series here instead (`result.loc[mask, "score"]
        # * w`) aligns on index and leaves every unmasked game NaN — which
        # sort_values then sinks to the bottom, so the only games surviving
        # into the top-N are the riskiest ones. That inverts the safety rule
        # exactly, and it is silent: the column looks populated.
        high_mult_threshold = game_vecs["risk_appetite"].quantile(HIGH_MAX_MULT_PERCENTILE)
        is_high_vol = result["vol_high"].to_numpy(dtype=bool)
        is_high_mult = (game_vecs["risk_appetite"] > high_mult_threshold).to_numpy(dtype=bool)

        adjusted = result["score"].to_numpy(dtype=float).copy()
        adjusted[is_high_vol] *= HIGH_VOLATILITY_DOWNWEIGHT
        adjusted[is_high_mult] *= HIGH_VOLATILITY_DOWNWEIGHT
        result["adjusted_score"] = adjusted

        # The note marks the rows that were actually adjusted. Scaling every
        # row by the same factor (as this did before) cannot reorder anything,
        # so a uniform downweight is not a safety measure at all.
        result["note"] = np.where(
            is_high_vol | is_high_mult,
            "downweighted: flagged player, high-volatility / high-multiplier game",
            "",
        )
    else:
        result["adjusted_score"] = result["score"]
        result["note"] = ""

    result = result.sort_values(by="adjusted_score", ascending=False).head(top_n).reset_index(drop=True)
    return result

def run_demo(data_dir: Path, top_n: int = 5) -> None:
    """Run a demo of the recommendation engine."""
    games = pd.read_csv(data_dir / "games.csv")
    sessions = pd.read_csv(data_dir / "sessions.csv")
    player_features = pd.read_csv(data_dir / "player_features_raw.csv")
    risk_table = pd.read_csv(data_dir / "player_risk.csv")

    game_vecs, theme_cols = build_game_vectors(games)
    player_vecs = build_player_vectors(sessions, games, player_features, theme_cols)

    sample_ids = player_features.groupby("persona_id")["player_id"].first()
    for persona_id, player_id in sample_ids.items():
        print(f"\nPersona {persona_id} (Player ID: {player_id}) recommendations:")
        recommendations = recommend_for_players(player_id, player_vecs, game_vecs, risk_table, top_n)
        print(f"Top {top_n} recommendations for {player_id} with persona {persona_id}:")
        print(recommendations[["game_id", "name", "vol_low", "vol_medium", "vol_high", "adjusted_score", "note"]])

def main():
    parser = argparse.ArgumentParser(description="Run the recommendation engine demo.")
    parser.add_argument("--data-dir", type=Path, required=True, help="Directory containing input CSV files.")
    parser.add_argument("--top-n", type=int, default=5, help="Number of top recommendations to return.")
    args = parser.parse_args()

    run_demo(args.data_dir, args.top_n)

if __name__ == "__main__":
    main()