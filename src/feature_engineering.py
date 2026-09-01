"""
Feature engineering & preprocessing.

Turns the raw simulator output (games.csv, players.csv, sessions.csv,
events.csv) into one row per player: a behavioural feature vector suitable
for clustering (cluster.py) and for the embedding model (embedding_model.py).

Two outputs are written to --out-dir:
    player_features_raw.csv     interpretable, unscaled — good for reading,
                                 for the dashboard, and for sanity-checking
                                 what a cluster actually represents.
    player_features_scaled.csv  the same features, standardised (zero mean,
                                 unit variance) — what the models consume.

persona_id is carried through as a label column but is NEVER used as a model
input — it's ground truth we can check clusters against, not a feature. Any
downstream script that fits a model must explicitly drop it.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# Columns a clustering/embedding model is allowed to see. Keeping this as an
# explicit list (rather than "everything except a blocklist") means adding a
# new raw column later can't silently leak into the model.
FEATURE_COLUMNS = [
    "n_sessions",
    "avg_bet_fraction_of_bankroll",
    "bet_std_fraction_of_bankroll",
    "avg_session_length",
    "avg_escalation_ratio",
    "max_consecutive_losses_mean",
    "pct_bankroll_depleted",
    "pct_voluntary_quit",
    "net_result_per_spin",
    "bonus_trigger_rate",
    "pref_low_volatility",
    "pref_medium_volatility",
    "pref_high_volatility",
    "theme_diversity",
]

LABEL_COLUMNS = ["player_id", "persona_id"]  # never used as a model input, only for sanity-

def load_and_preprocess_data(input_dir: Path) -> pd.DataFrame:
    """
    Load the raw simulator output and turn it into a single row per player.

    Args:
        input_dir: Path to the directory containing the raw CSV files.
    return:
        A DataFrame with one row per player, containing the features and labels.
    """
    return {
        "games": pd.read_csv(input_dir / "games.csv"),
        "players": pd.read_csv(input_dir / "players.csv"),
        "sessions": pd.read_csv(input_dir / "sessions.csv"),
        "events": pd.read_csv(input_dir / "events.csv"),
    }

def bonus_rate_per_player(sessions:pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    # Implementation for calculating bonus rate per player
    sessions_to_player = sessions.set_index("session_id")["player_id"]
    events["player_id"] = events["session_id"].map(sessions_to_player)
    return events.groupby("player_id")["is_bonus_triggered"].mean().rename("bonus_trigger_rate")

def volatility_preference_per_player(sessions: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the volatility preference for each player based on their sessions and games.
    """
    merged = sessions.merge(games[["game_id", "volatility"]], on="game_id", how="left")
    pref = (merged.groupby("player_id")["volatility"].value_counts(normalize=True).unstack(fill_value=0))
    pref = pref.rename(columns={"low": "pref_low_volatility",
                                "medium": "pref_medium_volatility",
                                "high": "pref_high_volatility"})
    for col in ["pref_low_volatility", "pref_medium_volatility", "pref_high_volatility"]:
        if col not in pref.columns:
            pref[col] = 0.0
    return pref[["pref_low_volatility", "pref_medium_volatility", "pref_high_volatility"]]


def build_player_features(raw: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Convert the raw DataFrames into a single row per player.

    Args:
        raw: Dictionary containing the raw DataFrames.
    return:
        A DataFrame with one row per player, containing the features and labels.
    """
    sessions = raw["sessions"].copy()
    games = raw["games"].copy()
    players = raw["players"].copy()
    events = raw["events"].copy()

    sessions["bet_fraction_of_bankroll"] = sessions["avg_bet"] / sessions["start_bankroll"].replace(0, np.nan)
    sessions["bet_std_fraction_of_bankroll"] = sessions["bet_std"] / sessions["start_bankroll"].replace(0, np.nan)
    sessions["net_result_per_spin"] = sessions["net_result"] / sessions["num_spins"].replace(0, np.nan)
    sessions["is_bankroll_depleted"] = sessions["end_reason"] == "bankroll_depleted"
    sessions["is_voluntary_quit"] = sessions["end_reason"] == "voluntary_quit"

    aggregated = sessions.groupby("player_id").agg(
        n_sessions=("session_id", "count"),
        avg_bet_fraction_of_bankroll=("bet_fraction_of_bankroll", "mean"),
        bet_std_fraction_of_bankroll=("bet_std_fraction_of_bankroll", "mean"),
        avg_session_length=("num_spins", "mean"),
        avg_escalation_ratio=("escalation_ratio", "mean"),
        max_consecutive_losses_mean=("max_consecutive_losses", "mean"),
        pct_bankroll_depleted=("is_bankroll_depleted", "mean"),
        pct_voluntary_quit=("is_voluntary_quit", "mean"),
        net_result_per_spin=("net_result_per_spin", "mean"),
        )

    aggregated = aggregated.join(bonus_rate_per_player(sessions, events))
    aggregated = aggregated.join(volatility_preference_per_player(sessions, games))

    out = players.set_index("player_id")[["persona_id"]].join(aggregated).reset_index()
    return out

def run(data_dir: Path, out_dir: Path) -> None:
    """
    Run the feature engineering pipeline.

    Args:
        data_dir: Path to the directory containing the raw CSV files.
        out_dir: Path to the directory where the output CSV files will be saved.
    """
    raw = load_and_preprocess_data(data_dir)
    features = build_player_features(raw)

    # Save unscaled features
    features.to_csv(out_dir / "player_features_raw.csv", index=False)

def main() -> None:
    parser = argparse.ArgumentParser(description="Feature engineering & preprocessing.")
    parser.add_argument("--data-dir", type=Path, required=True, help="Directory containing the raw CSV files.")
    parser.add_argument("--out-dir", type=Path, required=True, help="Directory where the output CSV files will be saved.")
    args = parser.parse_args()

    run(args.data_dir, args.out_dir)

if __name__ == "__main__":
    main()