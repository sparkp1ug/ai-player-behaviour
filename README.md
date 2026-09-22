# ai-player-behaviour

[![CI](https://github.com/sparkp1ug/ai-player-behaviour/actions/workflows/ci.yml/badge.svg)](https://github.com/sparkp1ug/ai-player-behaviour/actions/workflows/ci.yml)

**A recommender that has to learn while the players it serves are learning too.**

Most recommender projects score a fixed dataset offline. This one builds its
data by interaction: a live, RTP-calibrated slot machine; a population of
agents with different behavioural temperaments that learn their own tastes,
adapt their bet sizes, and decide for themselves whether to trust a
recommendation; and a contextual bandit that has to keep up while they change
underneath it.

Three policies face **identical agents on identically seeded machines**, so the
difference between them is caused by the policy and nothing else. Pooled over
6 seeds x 40 agents x 60 rounds:

| policy | satisfaction | accept rate | high-volatility recs to flagged players |
|---|---|---|---|
| random | 0.408 | 0.35 | 0.34 |
| static cosine | 0.420 | 0.41 | 0.04 |
| adaptive (LinUCB) | **0.435** | **0.47** | **0.00** |

![Learning curves: satisfaction, accept rate and safety-layer exposure over 60 rounds](docs/img/learning_curves.png)

Adaptive is ahead on 6/6 seeds, +0.016 satisfaction/session over static
(paired by seed, sd 0.010). The dip at round 30 is a deliberate taste shift —
every agent's preferences rotate at once — and the adaptive policy is the one
that recovers.

**Two caveats, stated here rather than buried.** The gap is small next to the
0.22 of headroom a perfect recommender would capture. And most of the
round-over-round rise is the agents learning their own tastes, not the
recommender improving — it happens under random recommendations too, which is
exactly why the baselines are in the table.

**What is being optimised matters.** The reward is session quality from the
player's side — taste fit, session length, bonus excitement — with penalties
for busting out and for chasing losses. It is never time-on-device or amount
wagered. Risk-flagged players are protected by a hard constraint that sits
*outside* the objective, because anything that must hold regardless of reward
should not be something a bandit is free to trade away.

## Try it

```bash
pip install -r requirements.txt

# play the machine yourself: enter spins, `b 5` changes bet, `q` quits
python src/slot_machine.py --play --game game_08 --bankroll 200 --bet 2

# check the machine pays what the catalogue claims
python src/slot_machine.py --audit --spins 200000

# 40 agents x 60 rounds x 3 policies, repeated over 6 seeds
python src/play_loop.py --agents-per-persona 8 --rounds 60 --seeds 7 11 23 31 47 59

# browse players, risk flags and recommendations
streamlit run dashboard/app.py
```

The arena's output is drop-in for the offline pipeline, which is the point of
sharing one schema between them:

```bash
python src/feature_engineering.py --data-dir data/arena --out-dir data/arena
python src/responsible_play.py --features-raw-path data/arena/player_features_raw.csv --out-dir data/arena
```

## Two loops

**Offline** — `generate_synthetic_data.py` rolls a dataset in one shot, and the
rest of the pipeline (features → clustering → risk scoring → cosine
recommendations) reads the CSVs it leaves behind.

**Online** — `slot_machine.py` turns the same payout maths into a live
environment; `agents.py` puts a population of learning players in front of it;
`play_loop.py` runs them round after round while `adaptive_recommender.py`
learns from what happens. The data is produced *by* the interaction, so a
recommender that changes its advice also changes the behaviour it will observe
next — something no offline replay can show you.

### How the comparison is set up

Every policy is run against a freshly built population using the same seed, so
agent temperaments, hidden theme preferences and the machine's RNG are all
identical across policies — the only thing that differs is how games get
chosen. Satisfaction is the session-quality reward defined in `agents.py`, in
[0, 1], and each run is repeated across six seeds with the comparison paired
seed by seed.

Three things make the target move, so a policy cannot memorise round 1 and
coast:

- agents learn their own game tastes from experience (Q-values)
- their trust in the recommender rises and falls with how well its advice lands
- bet boldness drifts with results — loss-chasing personas get bolder after
  going bust, cautious ones pull back

`--taste-shift-round` rotates every agent's theme preferences mid-run, which is
the dip at round 30 in the figure above.

A detail that turned out to matter: agents start knowing only 3 of the 15
games and rarely discover new ones alone. Without that, an agent left to
itself tries everything within a few rounds and learns its own favourite, and
no recommender can beat it. Making catalogue discovery scarce is what gives
the recommender something real to allocate.

## Tests

```bash
pip install -r requirements-dev.txt
ruff check . && pytest
```

25 tests, ~40 seconds. They check the things this project actually claims,
not just that the code runs:

- **`payout_scale` is exact.** The analytic solve is checked against an
  independent 200k-sample Monte Carlo estimate — two completely different
  computations that have to agree — and against the target RTP directly.
- **The machine pays what the catalogue says.** Realised RTP against the
  analytic expectation, plus tight Bernoulli bands on hit rate and bonus
  rate, which is what catches a mis-structured branch.
- **The safety layer.** A regression test for the inverted flagged-player
  path: every game keeps a real `adjusted_score`, flagged players see fewer
  high-volatility games, unflagged players are untouched.
- **The bandit beats its baselines.** Paired by seed. Adaptive vs random is
  asserted per seed (positive on 20/20 seeds when measured); adaptive vs
  static only on the mean, because over the same 20 seeds it was negative on
  one — asserting per seed there would buy a flaky test, not confidence.
- **The autoencoder's backward pass.** Against central finite differences,
  which is the only way to know a transposed matrix isn't quietly training
  anyway.

CI additionally runs every script end to end on a clean checkout, because
`data/` is gitignored and the unit tests build their own fixtures.

## Features
- Synthetic player behaviour dataset generator, with `payout_scale` solved
  exactly so each game's realised RTP matches its target
- Behaviour feature engineering and preprocessing
- Clustering using scikit-learn (K-means, PCA visualisation)
- Behavioural embedding model: NumPy autoencoder, backward pass verified by
  gradient check
- Cosine-similarity recommendation engine with a responsible-play safety layer
- Playable slot machine environment sharing the dataset's payout maths
- Population of learning AI agents that generate data by interacting
- Adaptive recommender (hybrid LinUCB contextual bandit) learning online,
  benchmarked against random and static-cosine baselines
- Interactive dashboard (Streamlit): player profiles, risk flags,
  recommendations, arena results

## Purpose

This project is intended as a portfolio piece to demonstrate skills in:
- Machine learning engineering
- Behaviour modelling
- Recommendation systems
- Data visualisation
- Python backend development
- Interactive dashboards

It is **not** intended for deployment in any gambling environment and complies
with responsible AI and safe modelling principles.