from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

RISK_COMPONENTS = {
    "avg_escalation_ratio": 0.35,        # bets grow a lot after a loss
    "avg_bet_fraction_of_bankroll": 0.20,             # betting a large share of bankroll
    "pct_bankroll_depleted": 0.25,        # sessions that end by going bust
    "max_consecutive_losses_mean": 0.20,  # rides out long losing streaks
}

def _zscore(series: pd.Series) -> pd.Series:
    std = series.std()
    if std == 0 or pd.isna(std):
        return series * 0.0
    return (series - series.mean()) / std

def score(features_raw: pd.DataFrame, flag_quantile: float = 0.90) -> pd.DataFrame:
    df = features_raw.copy()
    df["avg_escalation_ratio"] = df["avg_escalation_ratio"].fillna(df["avg_escalation_ratio"].median())

    z_components = pd.DataFrame({
        col: _zscore(df[col]) for col in RISK_COMPONENTS
    })
    weights = pd.Series(RISK_COMPONENTS)
    risk_score = (z_components * weights).sum(axis=1)

    threshold = risk_score.quantile(flag_quantile)

    result = df[["player_id", "persona_id"]].copy()
    for col in RISK_COMPONENTS:
        result[f"z_{col}"] = z_components[col].values
    result["risk_score"] = risk_score.values
    result["risk_flag"] = (risk_score >= threshold).values
    return result

def run(features_raw_path: Path, out_dir: Path, flag_quantile: float) -> None:
    features_raw = pd.read_csv(features_raw_path)
    result = score(features_raw, flag_quantile)

    out_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(out_dir / "player_risk.csv", index=False)

    print(f"flagged {result['risk_flag'].sum()} / {len(result)} players "
          f"(top {100*(1-flag_quantile):.0f}% by risk score)")
    print("\nflag rate by persona (validation only — persona_id isn't used to compute the score):")
    print(result.groupby("persona_id")["risk_flag"].mean().round(3))
    print(f"\nwrote {out_dir/'player_risk.csv'}")

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features-raw-path", type=Path, default=Path("data/player_features_raw.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("data"))
    parser.add_argument("--flag-quantile", type=float, default=0.90,
                         help="Players at/above this quantile of risk_score are flagged.")
    args = parser.parse_args()

    run(args.features_raw_path, args.out_dir, args.flag_quantile)

if __name__ == "__main__":
    main()