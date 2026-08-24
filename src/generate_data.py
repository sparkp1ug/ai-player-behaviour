import numpy as np
import pandas as pd
from pathlib import Path

def generate_synthetic_players(n=10000, seed=42):
    np.random.seed(seed)

    # Volatility preference: 0 = low, 1 = medium, 2 = high
    volatility_pref = np.random.choice([0, 1, 2], size=n, p=[0.4, 0.4, 0.2])

    # Session length in minutes
    session_length = np.random.gamma(shape=2.0, scale=15.0, size=n)

    # Spin frequency per minute
    spin_frequency = np.random.normal(loc=30, scale=10, size=n).clip(5, 80)

    # Bet size category: 0 = low, 1 = medium, 2 = high
    bet_size_category = np.random.choice([0, 1, 2], size=n, p=[0.6, 0.3, 0.1])

    # Feature engagement (0–1 scale)
    feature_engagement = np.random.beta(a=2, b=5, size=n)

    # Exploration rate (0–1 scale)
    exploration_rate = np.random.beta(a=3, b=3, size=n)

    # Reward sensitivity (0–1 scale)
    reward_sensitivity = np.random.beta(a=2, b=4, size=n)

    df = pd.DataFrame({
        "volatility_pref": volatility_pref,
        "session_length": session_length,
        "spin_frequency": spin_frequency,
        "bet_size_category": bet_size_category,
        "feature_engagement": feature_engagement,
        "exploration_rate": exploration_rate,
        "reward_sensitivity": reward_sensitivity,
    })

    return df

if __name__ == "__main__":
    df = generate_synthetic_players()
    output_path = Path("data/players.parquet")
    output_path.parent.mkdir(exist_ok=True)
    df.to_parquet(output_path)
    print(f"Generated synthetic dataset → {output_path}")
