"""
The arena: a population of AI agents plays the live slot machine, round after
round, while the recommender learns from what happens.

One round = every agent gets one recommendation, decides whether to take it,
plays a session, and reports how satisfying that session was. The recommender
updates on the result and tries again next round. Nothing here is replayed
from a fixed dataset — the data is created by the interaction, which is the
point: a recommender that changes its advice changes the behaviour it will
observe next, and offline evaluation cannot show you that.

Three policies run over identical agent populations and identical machine
seeds, so the only difference between them is how games get chosen:

    random    uniform over the catalogue — the floor
    static    the existing cosine engine, frozen — good cold start, no learning
    adaptive  hybrid LinUCB warm-started from that same cosine prior

Because the agents themselves adapt (bet boldness drifts, trust in the
recommender rises and falls) and tastes can be shifted mid-run with
--taste-shift-round, the target is moving. A policy that "learned" by
memorising round 1 loses; the comparison is designed to make that visible.

Outputs (under --out-dir):
    sessions.csv           one row per session, same schema as data/sessions.csv
    events.csv             one row per spin for the adaptive arena
    players.csv            the agent roster
    recommendations.csv    per round: what was recommended, taken, and rewarded
    learning_curves.png    satisfaction / accept rate / safety over time

Usage:
    python src/play_loop.py --agents 40 --rounds 50 --seed 7
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import responsible_play  # noqa: E402
from adaptive_recommender import (  # noqa: E402
    AdaptiveRecommender,
    RandomRecommender,
    StaticCosineRecommender,
)
from agents import PlayerAgent, build_agents  # noqa: E402
from recommendation_engine import build_game_vectors  # noqa: E402
from schema import DEFAULT_PERSONAS, EndReason  # noqa: E402
from slot_machine import SlotMachine  # noqa: E402

POLICIES = ["random", "static", "adaptive"]
RISK_WARMUP_ROUNDS = 3       # need some history before risk scores mean anything
RISK_FLAG_QUANTILE = 0.90    # same default as responsible_play.py


def _make_policy(name: str, game_ids: list[str], game_vecs: pd.DataFrame, seed: int):
    if name == "random":
        return RandomRecommender(game_ids, seed=seed)
    if name == "static":
        return StaticCosineRecommender(game_ids, game_vecs, seed=seed)
    if name == "adaptive":
        return AdaptiveRecommender(game_ids, game_vecs, seed=seed)
    raise ValueError(f"unknown policy {name!r}")


def risk_flags(agents: list[PlayerAgent], round_idx: int) -> dict[str, bool]:
    """Score the live population with the existing responsible-play model.

    Same weights, same z-scoring, same flag quantile as the offline version —
    the only difference is that the features come from agents' running stats
    instead of a CSV, so a player's flag can appear (or clear) mid-run.
    """
    if round_idx < RISK_WARMUP_ROUNDS:
        return {}
    contexts = [a.context() for a in agents]
    frame = pd.DataFrame({
        "player_id": [a.agent_id for a in agents],
        "persona_id": [a.persona.persona_id for a in agents],
        "avg_escalation_ratio": [c["avg_escalation_ratio"] for c in contexts],
        "avg_bet_fraction_of_bankroll": [c["avg_bet_fraction_of_bankroll"] for c in contexts],
        "pct_bankroll_depleted": [c["pct_bankroll_depleted"] for c in contexts],
        "max_consecutive_losses_mean": [c["max_consecutive_losses_mean"] for c in contexts],
    })
    scored = responsible_play.score(frame, flag_quantile=RISK_FLAG_QUANTILE)
    return dict(zip(scored["player_id"], scored["risk_flag"], strict=True))


def run_policy(
    policy_name: str,
    machine_games,
    themes: list[str],
    n_per_persona: int,
    rounds: int,
    seed: int,
    taste_shift_round: int | None,
    collect_events: bool,
) -> dict[str, pd.DataFrame]:
    """Run one full arena. Agents and machine are re-seeded identically for
    every policy, so any difference in the results is down to the policy."""
    machine = SlotMachine(machine_games, seed=seed)
    agents = build_agents(DEFAULT_PERSONAS, n_per_persona, themes,
                          machine.game_ids, np.random.default_rng(seed))
    for agent in agents:
        agent.bind_catalogue(machine)

    game_vecs, _ = build_game_vectors(pd.DataFrame([vars(g) for g in machine_games]).assign(
        volatility=[g.volatility.value for g in machine_games]))
    policy = _make_policy(policy_name, machine.game_ids, game_vecs, seed)
    shift_rng = np.random.default_rng(seed + 1)

    rec_rows: list[dict] = []
    session_rows: list[dict] = []
    event_rows: list[dict] = []
    high_vol = {g.game_id for g in machine_games if g.volatility.value == "high"}

    for round_idx in range(rounds):
        if taste_shift_round is not None and round_idx == taste_shift_round:
            for agent in agents:
                agent.shift_tastes(themes, shift_rng)

        flags = risk_flags(agents, round_idx)

        for agent in agents:
            context = agent.context()
            flagged = bool(flags.get(agent.agent_id, False))

            recommended = policy.recommend(context, agent.agent_id, flagged)
            game_id, accepted = agent.consider(recommended, machine.game_ids)

            outcome = agent.play(machine, game_id)
            agent.learn(game_id, outcome, accepted)
            # The reward is only observed for what was actually played, so
            # that is the arm we update — including when the agent ignored us.
            policy.update(context, agent.agent_id, game_id, outcome.satisfaction)

            rec_rows.append({
                "round": round_idx,
                "policy": policy_name,
                "seed": seed,
                "player_id": agent.agent_id,
                "persona_id": agent.persona.persona_id,
                "recommended_game": recommended,
                "played_game": game_id,
                "accepted": accepted,
                "risk_flagged": flagged,
                "recommended_high_volatility": recommended in high_vol,
                "satisfaction": round(outcome.satisfaction, 4),
                "trust": round(agent.trust, 4),
                "boldness": round(agent.boldness, 4),
                "bankroll": round(agent.bankroll, 2),
                "net_result": outcome.session.net_result,
                "num_spins": outcome.session.num_spins,
                "busted": outcome.session.end_reason == EndReason.BANKROLL_DEPLETED,
            })

            row = vars(outcome.session).copy()
            row["end_reason"] = row["end_reason"].value
            row["round"] = round_idx
            session_rows.append(row)

            if collect_events:
                event_rows.extend(vars(e) for e in outcome.events)

    players = pd.DataFrame([{
        "player_id": a.agent_id,
        "persona_id": a.persona.persona_id,
        "starting_bankroll": a.starting_bankroll,
        "deposits": a.deposits,
        "final_bankroll": round(a.bankroll, 2),
        "final_trust": round(a.trust, 4),
        "final_boldness": round(a.boldness, 4),
        "favourite_theme": max(a.theme_affinity, key=a.theme_affinity.get),
    } for a in agents])

    return {
        "policy": policy,
        "agents": agents,
        "recommendations": pd.DataFrame(rec_rows),
        "sessions": pd.DataFrame(session_rows),
        "events": pd.DataFrame(event_rows),
        "players": players,
        "_machine_rtp": machine.realised_rtp,
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def summarise(results: dict[str, dict], rounds: int) -> pd.DataFrame:
    """Per-policy scoreboard, pooled over seeds.

    Read it by comparing *policies*, not by reading one policy's numbers on
    their own. Every policy faces the same agents with the same seeds, so a
    difference between rows is caused by the policy. The first-quarter to
    last-quarter lift, by contrast, is mostly the agents learning their own
    tastes — it rises even under random recommendations, which is exactly why
    a baseline row is here.
    """
    quarter = max(rounds // 4, 1)
    rows = []
    for name, res in results.items():
        recs = res["recommendations"]
        per_seed = recs.groupby("seed")["satisfaction"].mean()
        early = recs[recs["round"] < quarter]
        late = recs[recs["round"] >= rounds - quarter]
        flagged = recs[recs["risk_flagged"]]
        rows.append({
            "policy": name,
            "satisfaction": round(recs["satisfaction"].mean(), 4),
            "sat_sd_across_seeds": round(float(per_seed.std(ddof=0)), 4),
            "sat_first_q": round(early["satisfaction"].mean(), 4),
            "sat_last_q": round(late["satisfaction"].mean(), 4),
            "lift": round(late["satisfaction"].mean() - early["satisfaction"].mean(), 4),
            "accept_rate": round(recs["accepted"].mean(), 4),
            "accept_last_q": round(late["accepted"].mean(), 4),
            "bust_rate": round(recs["busted"].mean(), 4),
            "high_vol_to_flagged": (
                round(flagged["recommended_high_volatility"].mean(), 4) if len(flagged) else float("nan")
            ),
            "net_per_spin": round(
                recs["net_result"].sum() / max(recs["num_spins"].sum(), 1), 4),
            "sessions": len(recs),
        })
    return pd.DataFrame(rows).set_index("policy")


def plot_learning_curves(results: dict[str, dict], out_path: Path,
                         taste_shift_round: int | None) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    window = 5

    panels = [
        ("satisfaction", "Mean session satisfaction", "reward the recommender is optimising"),
        ("accepted", "Recommendation accept rate", "agents choose whether to follow advice"),
        ("recommended_high_volatility", "High-volatility share, flagged players",
         "safety layer — lower is better, not learned"),
    ]

    for ax, (column, title, subtitle) in zip(axes, panels, strict=True):
        for name, res in results.items():
            recs = res["recommendations"]
            if column == "recommended_high_volatility":
                recs = recs[recs["risk_flagged"]]
                if recs.empty:
                    continue
            # mean over agents and seeds, then smoothed
            series = recs.groupby("round")[column].mean().rolling(window, min_periods=1).mean()
            ax.plot(series.index, series.values, label=name, linewidth=1.8)
        if taste_shift_round is not None:
            ax.axvline(taste_shift_round, color="grey", linestyle=":", linewidth=1.2)
            ax.text(taste_shift_round, ax.get_ylim()[1], " taste shift",
                    color="grey", fontsize=8, va="top")
        ax.set_title(f"{title}\n{subtitle}", fontsize=10)
        ax.set_xlabel("round")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel(f"rolling mean ({window} rounds)")
    axes[0].legend(frameon=False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def show_learned_recommendations(result: dict, n: int = 5) -> None:
    """Print what the trained bandit now suggests for two contrasting players.

    The scoreboard says the policy is better on average; this says what
    "better" looks like for one person, which is the thing anyone reviewing a
    recommender actually wants to see.
    """
    policy, agents = result["policy"], result["agents"]
    flags = risk_flags(agents, RISK_WARMUP_ROUNDS)
    flagged = next((a for a in agents if flags.get(a.agent_id)), None)
    unflagged = next((a for a in agents if not flags.get(a.agent_id)), None)

    for agent, label in [(unflagged, "not flagged"), (flagged, "risk-flagged")]:
        if agent is None:
            continue
        top = policy.top_n(agent.context(), agent.agent_id,
                           risk_flag=bool(flags.get(agent.agent_id)), n=n)
        favourite = max(agent.theme_affinity, key=agent.theme_affinity.get)
        print(f"\nlearned recommendations for {agent.agent_id} "
              f"({agent.persona.persona_id}, {label})")
        print(f"  hidden truth the model never saw: favourite theme = {favourite}, "
              f"trust = {agent.trust:.2f}, boldness = {agent.boldness:.2f}")
        print("  " + top.to_string(index=False).replace("\n", "\n  "))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--games-csv", type=Path, default=Path("data/games.csv"))
    parser.add_argument("--agents-per-persona", type=int, default=8,
                        help=f"{len(DEFAULT_PERSONAS)} personas, so total agents = 5x this")
    parser.add_argument("--rounds", type=int, default=50, help="sessions played by each agent")
    parser.add_argument("--seeds", type=int, nargs="+", default=[7],
                        help="run the whole arena once per seed and pool the "
                             "results; a single run's policy gap is well inside "
                             "seed-to-seed noise, so pass 3-6 seeds before "
                             "believing any difference")
    parser.add_argument("--taste-shift-round", type=int, default=None,
                        help="rotate every agent's theme preferences at this round "
                             "(default: halfway; pass -1 to disable)")
    parser.add_argument("--policies", nargs="+", default=POLICIES, choices=POLICIES)
    parser.add_argument("--out-dir", type=Path, default=Path("data/arena"))
    parser.add_argument("--no-events", action="store_true",
                        help="skip the per-spin events.csv (it is by far the biggest file)")
    args = parser.parse_args()

    shift = args.rounds // 2 if args.taste_shift_round is None else args.taste_shift_round
    shift = None if shift is not None and shift < 0 else shift

    machine_games = SlotMachine.from_csv(args.games_csv).games
    games = list(machine_games.values())
    themes = sorted({g.theme for g in games})

    n_agents = args.agents_per_persona * len(DEFAULT_PERSONAS)
    print(f"{n_agents} agents x {args.rounds} rounds x {len(args.policies)} policies "
          f"= {n_agents * args.rounds * len(args.policies):,} sessions to simulate")
    if shift is not None:
        print(f"theme preferences shift at round {shift}")

    results: dict[str, dict] = {}
    for policy_name in args.policies:
        t0 = time.time()
        runs = []
        for seed in args.seeds:
            runs.append(run_policy(
                policy_name=policy_name,
                machine_games=games,
                themes=themes,
                n_per_persona=args.agents_per_persona,
                rounds=args.rounds,
                seed=seed,
                taste_shift_round=shift,
                # Only the first seed's spin-level data is kept; every seed's
                # session summaries and recommendation log are pooled.
                collect_events=(policy_name == "adaptive" and not args.no_events
                                and seed == args.seeds[0]),
            ))
        results[policy_name] = {
            "policy": runs[0]["policy"],
            "agents": runs[0]["agents"],
            "recommendations": pd.concat([r["recommendations"] for r in runs], ignore_index=True),
            "sessions": runs[0]["sessions"],
            "events": runs[0]["events"],
            "players": runs[0]["players"],
            "_machine_rtp": float(np.mean([r["_machine_rtp"] for r in runs])),
        }
        spins = results[policy_name]["recommendations"]["num_spins"].sum()
        print(f"  {policy_name:<9} {spins:>9,} spins  "
              f"machine RTP {results[policy_name]['_machine_rtp']:.2%}  "
              f"{time.time() - t0:5.1f}s")

    summary = summarise(results, args.rounds)
    print("\n" + summary.to_string())


    if "adaptive" in results:
        adaptive = results["adaptive"]["recommendations"].groupby("seed")["satisfaction"].mean()
        for other in [p for p in args.policies if p != "adaptive"]:
            base = results[other]["recommendations"].groupby("seed")["satisfaction"].mean()
            diff = (adaptive - base)   # paired by seed: same agents, same machine
            wins = int((diff > 0).sum())
            print(f"adaptive vs {other:<7} {diff.mean():+.4f} satisfaction/session "
                  f"(paired by seed, sd {diff.std(ddof=0):.4f}, "
                  f"adaptive ahead on {wins}/{len(diff)} seeds)")
        if len(args.seeds) < 3 and len(args.policies) > 1:
            print("  ^ one or two seeds is not evidence; re-run with --seeds 7 11 23 31 47 59")
        show_learned_recommendations(results["adaptive"])

    # The adaptive arena is the dataset worth keeping: it is the one produced
    # by a system that was actually learning while the data was generated.
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    main_policy = "adaptive" if "adaptive" in results else args.policies[0]
    res = results[main_policy]

    res["sessions"].to_csv(out / "sessions.csv", index=False)
    res["players"].to_csv(out / "players.csv", index=False)
    pd.concat([r["recommendations"] for r in results.values()]).to_csv(
        out / "recommendations.csv", index=False)
    summary.to_csv(out / "summary.csv")
    pd.read_csv(args.games_csv).to_csv(out / "games.csv", index=False)
    if not res["events"].empty:
        res["events"].to_csv(out / "events.csv", index=False)

    plot_learning_curves(results, out / "learning_curves.png", shift)

    print(f"\nwrote {out}/sessions.csv ({len(res['sessions']):,} rows), players.csv, "
          f"recommendations.csv, summary.csv"
          + (f", events.csv ({len(res['events']):,} rows)" if not res["events"].empty else "")
          + ", learning_curves.png")
    print(f"the arena output is drop-in for the offline pipeline:\n"
          f"  python src/feature_engineering.py --data-dir {out} --out-dir {out}")


if __name__ == "__main__":
    main()
