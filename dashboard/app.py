"""
Streamlit dashboard: player profiles, recommendations, and the live arena.

Two tabs, matching the project's two halves:

  Players — pick a player, see the behaviour the model actually observes,
            their cluster, whether the responsible-play model flagged them,
            and what the cosine engine recommends as a result.
  Arena   — the online half: how the adaptive recommender compares with its
            baselines when a population of learning agents plays the live
            slot machine.

Each tab degrades on its own if the data behind it hasn't been generated —
deliberately not via st.stop(), which halts the whole script and would blank
the other tab too.

Run from the repo root:  streamlit run dashboard/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# src/ modules import each other flatly (`from schema import ...`), so src/
# itself has to be importable, not just the repo root.
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from recommendation_engine import (  # noqa: E402
    build_game_vectors,
    build_player_vectors,
    recommend_for_players,
)

DATA = _ROOT / "data"
ARENA = DATA / "arena"
IMG = _ROOT / "docs" / "img"

st.set_page_config(page_title="Player Behaviour Explorer", layout="wide")


@st.cache_data
def load_offline() -> dict[str, pd.DataFrame] | None:
    """Everything the offline pipeline writes. None if it hasn't run."""
    required = ["games.csv", "sessions.csv", "player_features_raw.csv"]
    if not all((DATA / f).exists() for f in required):
        return None
    out = {name.removesuffix(".csv"): pd.read_csv(DATA / name) for name in required}
    for optional in ["player_risk.csv", "clustered_data.csv", "player_embeddings.csv"]:
        if (DATA / optional).exists():
            out[optional.removesuffix(".csv")] = pd.read_csv(DATA / optional)
    return out


@st.cache_data
def load_arena() -> dict[str, pd.DataFrame] | None:
    if not (ARENA / "summary.csv").exists():
        return None
    out = {"summary": pd.read_csv(ARENA / "summary.csv")}
    for optional in ["recommendations.csv", "sessions.csv"]:
        if (ARENA / optional).exists():
            out[optional.removesuffix(".csv")] = pd.read_csv(ARENA / optional)
    return out


def render_players(data: dict[str, pd.DataFrame]) -> None:
    features = data["player_features_raw"]
    risk = data.get("player_risk")
    clusters = data.get("clustered_data")

    # Selected by player_id, not by row number: the ids are strings, and a
    # positional index silently means a different player whenever the dataset
    # is regenerated.
    personas = sorted(features["persona_id"].unique())
    chosen = st.sidebar.selectbox("Filter by persona", ["(all)"] + personas)
    pool = features if chosen == "(all)" else features[features.persona_id == chosen]

    if risk is not None and st.sidebar.checkbox("Only risk-flagged players", value=False):
        pool = pool[pool.player_id.isin(set(risk.loc[risk.risk_flag, "player_id"]))]

    if pool.empty:
        st.info("No players match that filter.")
        return

    player_id = st.sidebar.selectbox("Player", pool["player_id"].tolist())
    top_n = st.sidebar.slider("Recommendations to show", 3, 10, 5)

    row = features.set_index("player_id").loc[player_id]
    has_risk = risk is not None and player_id in set(risk.player_id)
    is_flagged = bool(risk.set_index("player_id").loc[player_id, "risk_flag"]) if has_risk else False

    left, right = st.columns(2)

    with left:
        st.subheader("Profile")
        cols = st.columns(3)
        cols[0].metric("Persona", row["persona_id"])
        cols[1].metric("Sessions", int(row["n_sessions"]))
        if clusters is not None and player_id in set(clusters.player_id):
            cols[2].metric("Cluster", int(clusters.set_index("player_id").loc[player_id, "cluster"]))

        if is_flagged:
            st.error(
                "Risk-flagged by `responsible_play.py` — top 10% of the population "
                "by risk score. High-volatility games are downweighted below."
            )
        elif has_risk:
            st.success("Not risk-flagged.")

        st.caption(
            "persona_id is ground truth for validation only — no model here takes "
            "it as an input."
        )
        st.dataframe(row.drop(labels=["persona_id"]).rename("value").to_frame())

        if has_risk:
            st.subheader("Risk components")
            rrow = risk.set_index("player_id").loc[player_id]
            st.caption(f"weighted risk score: {rrow['risk_score']:.3f}")
            st.bar_chart(rrow[[c for c in risk.columns if c.startswith("z_")]].astype(float))

    with right:
        st.subheader("Recommendations")
        game_vecs, theme_cols = build_game_vectors(data["games"])
        player_vecs = build_player_vectors(data["sessions"], data["games"], features, theme_cols)
        recs = recommend_for_players(player_id, player_vecs, game_vecs, risk, top_n=top_n)

        display = recs[["game_id", "name", "score", "adjusted_score"]].copy()
        display.insert(2, "volatility", [
            "high" if h else ("medium" if m else "low")
            for h, m in zip(recs["vol_high"], recs["vol_medium"], strict=True)
        ])
        st.dataframe(display, hide_index=True)

        if is_flagged:
            st.caption(
                "`score` is the raw cosine similarity, `adjusted_score` is after the "
                "safety downweight. It is a soft penalty (x0.25 for a high-volatility, "
                "high-multiplier game), not a filter — a game with a large enough "
                "similarity lead can still rank first."
            )
        else:
            st.caption("No adjustment applied: `adjusted_score` equals `score`.")


def render_arena(arena: dict[str, pd.DataFrame]) -> None:
    st.subheader("Policy comparison")
    st.caption(
        "Every policy faces the same agents on the same machine seeds, so a "
        "difference between rows is caused by the policy. Satisfaction is the "
        "session-quality reward defined in agents.py, in [0, 1]."
    )
    st.dataframe(arena["summary"], hide_index=True)

    if (IMG / "learning_curves.png").exists():
        st.image(str(IMG / "learning_curves.png"))

    recs = arena.get("recommendations")
    if recs is None:
        return

    st.subheader("Satisfaction by round")
    window = st.slider("Rolling window (rounds)", 1, 15, 5)
    curve = (
        recs.groupby(["round", "policy"])["satisfaction"].mean()
        .unstack("policy").rolling(window, min_periods=1).mean()
    )
    st.line_chart(curve)
    st.caption(
        "Most of the round-over-round rise is the agents learning their own tastes — "
        "it happens under random recommendations too, which is why the baselines "
        "are here."
    )


st.title("Player Behaviour Explorer")
players_tab, arena_tab = st.tabs(["Players", "Arena (online learning)"])

with players_tab:
    offline = load_offline()
    if offline is None:
        st.warning(
            "No dataset found. Generate one first:\n\n"
            "```bash\n"
            "python src/generate_synthetic_data.py --seed 42 --out-dir data\n"
            "python src/feature_engineering.py --data-dir data --out-dir data\n"
            "python src/responsible_play.py\n"
            "python src/clustering.py\n"
            "```"
        )
    else:
        render_players(offline)

with arena_tab:
    arena_data = load_arena()
    if arena_data is None:
        st.warning(
            "No arena results found. Run:\n\n"
            "```bash\n"
            "python src/play_loop.py --agents-per-persona 8 --rounds 60 "
            "--seeds 7 11 23 31 47 59\n"
            "```"
        )
    else:
        render_arena(arena_data)
