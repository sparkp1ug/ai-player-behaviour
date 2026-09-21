# Architecture Overview

This project models synthetic player behaviour, clusters engagement styles, and
recommends suitable experiences using a similarity-based engine.

## Components

### 1. Synthetic Data Generator
Creates realistic player behaviour profiles:
- volatility preference
- session length
- spin frequency
- bet size category
- feature engagement
- exploration rate
- reward sensitivity

### 2. Behaviour Modelling
- StandardScaler for feature normalization
- PCA for dimensionality reduction
- K-means clustering for behaviour grouping

### 3. Experience Embedding
Each experience is represented by:
- volatility
- pace
- mechanics
- complexity
- audiovisual intensity
- reward structure

### 4. Recommendation Engine
Uses cosine similarity between:
- player PCA vector
- experience PCA vectors

### 5. Live Slot Machine (`src/slot_machine.py`)
A playable environment rather than a one-shot generator. Imports the
RTP-calibrated spin maths from `generate_synthetic_data.py` instead of
reimplementing it, so a live spin and an offline spin come from the same
distribution. Holds the catalogue and the RNG but no bankroll — the player
owns the money, which is what lets one machine serve a whole population.
`--audit` reports realised vs target RTP per game; `--play` is a terminal
front-end for a human.

### 6. AI Agent Population (`src/agents.py`)
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

### 7. Adaptive Recommender (`src/adaptive_recommender.py`)
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

### 8. Arena (`src/play_loop.py`)
Runs every policy over identical agent populations and machine seeds, pools
results across seeds, writes sessions/events/players/recommendations CSVs in
the same schema as the offline dataset, and plots satisfaction, accept rate
and safety-layer exposure over time. Supports a mid-run taste shift so the
recovery behaviour of a learning policy is visible.

### 9. Dashboard
Streamlit UI for:
- player profile inspection
- cluster visualization
- recommended experiences
