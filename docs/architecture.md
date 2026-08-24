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

### 5. Dashboard
Streamlit UI for:
- player profile inspection
- cluster visualization
- recommended experiences
