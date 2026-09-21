# Architecture Overview

This project models synthetic player behaviour, clusters engagement styles, and
recommends suitable experiences using a similarity-based engine.

## Components

### 1. Synthetic Data Generator (`src/generate_synthetic_data.py`)
Simulates five behavioural personas (`src/schema.py`) spinning through
sessions on a generated game catalogue. Each game's `payout_scale` is solved
exactly so that E[payout/bet] equals its target RTP, which makes RTP a real
driver of the simulation rather than a decorative field. Personas differ in
bet escalation after a loss, decay after a win, bankroll stop fraction, and
how a losing streak changes their quit probability — the loss chaser's is
negative, so it plays *through* losses.

Writes `games.csv`, `players.csv`, `sessions.csv` and `events.csv` (one row
per spin).

### 2. Feature Engineering (`src/feature_engineering.py`)
Aggregates the raw tables into one row per player: 13 behavioural features
(bet fraction of bankroll, session length, escalation ratio, bust rate, bonus
trigger rate, volatility preferences, and so on), written both unscaled for
reading and StandardScaler'd for models. `persona_id` is carried as a label
and never used as a model input — the explicit `FEATURE_COLUMNS` allowlist is
what stops a new column leaking in later.

### 3. Behaviour Modelling (`src/clustering.py`, `src/embedding_model.py`)
Clustering picks k by silhouette score, fits K-means, and projects to 2-D with
PCA for the plot only — the clustering itself runs on the full scaled feature
space, not on the PCA projection. The persona-vs-cluster crosstab is a sanity
check, not a training signal.

The embedding model is a from-scratch NumPy autoencoder (one tanh hidden layer
as the bottleneck, linear decoder, full-batch gradient descent) whose backward
pass is verified against finite differences. It does **not** beat PCA on
reconstruction — 4.040 against 4.020 at the same width — and says so; its
actual merit is a bounded embedding, since tanh stops any outlier dominating a
distance metric computed in that space.

### 4. Recommendation Engine (`src/recommendation_engine.py`)
Cosine similarity in an interpretable space, **not** a PCA space: games are
one-hot theme and volatility plus `bonus_affinity` (bonus frequency) and
`risk_appetite` (max multiplier), and players get the matching vector from
their play history and behavioural features. The volatility one-hots are
deliberately excluded from the similarity itself.

Players flagged by the responsible-play model get high-volatility and
high-multiplier games downweighted. This is a soft penalty (x0.5 per rule,
compounding), not a filter — a game with a large enough similarity lead can
still rank first.

### 5. Responsible Play (`src/responsible_play.py`)
Weighted z-scores over four at-risk signals — escalation after losses, bet
size relative to bankroll, bust rate, and tolerance for long losing streaks —
flagging the top decile. Validated by checking the flag rate per persona: it
fires on loss chasers and essentially nobody else, without ever seeing
`persona_id`.

### 6. Live Slot Machine (`src/slot_machine.py`)
A playable environment rather than a one-shot generator. Imports the
RTP-calibrated spin maths from `generate_synthetic_data.py` instead of
reimplementing it, so a live spin and an offline spin come from the same
distribution. Holds the catalogue and the RNG but no bankroll — the player
owns the money, which is what lets one machine serve a whole population.
`--audit` reports realised vs target RTP per game; `--play` is a terminal
front-end for a human.

### 7. AI Agent Population (`src/agents.py`)
Each agent wraps a schema persona (fixed temperament: bet escalation, quit
behaviour, bankroll limits) plus three things that adapt:
- **game taste** — per-game Q-values updated from session satisfaction
- **trust** — whether to follow the recommender, learned from whether past
  advice beat the agent's own running expectation
- **boldness** — a bet-size multiplier that drifts with results (chasers get
  bolder after busting), which makes the population non-stationary

Agents only know a handful of games at the start and rarely discover new ones
alone, so catalogue discovery is the scarce resource the recommender
allocates. Reward is session quality from the player's point of view, with
penalties for busting out and for loss-chasing — never engagement or spend.

### 8. Adaptive Recommender (`src/adaptive_recommender.py`)
Hybrid LinUCB contextual bandit:
- ridge regression per game over the player's observable behaviour vector,
  plus an optimism bonus for under-explored games (the part that generalises)
- a shrunk per-player residual on top (the part that personalises)
- warm-started from the static cosine engine's similarity, whose weight decays
  as real evidence arrives
- a non-learned safety layer reusing `responsible_play.py`'s flags and
  `recommendation_engine.py`'s downweight constant

`RandomRecommender` and `StaticCosineRecommender` are the baselines that turn
"it learns" into a measurement.

### 9. Arena (`src/play_loop.py`)
Runs every policy over identical agent populations and machine seeds, pools
results across seeds, writes sessions/events/players/recommendations CSVs in
the same schema as the offline dataset, and plots satisfaction, accept rate
and safety-layer exposure over time. Supports a mid-run taste shift so the
recovery behaviour of a learning policy is visible.

### 10. Dashboard (`dashboard/app.py`)
Streamlit, two tabs. Players: profile, cluster, responsible-play flag with the
z-scores behind it, and the resulting recommendations. Arena: the policy
comparison table, learning curves, and satisfaction by round.
