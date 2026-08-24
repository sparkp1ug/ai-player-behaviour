import numpy as np
import pandas as pd
import json
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity


# ---------------------------------------------------------
# Load player models (trained during clustering)
# ---------------------------------------------------------
def load_player_models():
    scaler = joblib.load("models/scaler.pkl")
    pca = joblib.load("models/pca.pkl")
    return scaler, pca


# ---------------------------------------------------------
# Load a single player's raw feature vector
# ---------------------------------------------------------
def load_player(player_id):
    df = pd.read_parquet("data/players.parquet")
    return df.iloc[player_id].values.reshape(1, -1)


# ---------------------------------------------------------
# Load experience definitions
# ---------------------------------------------------------
def load_experiences():
    with open("data/experiences.json") as f:
        return json.load(f)


# ---------------------------------------------------------
# Convert experience JSON into numeric vectors
# ---------------------------------------------------------
def embed_experiences(experiences):
    vectors = []
    for exp in experiences:
        vec = [
            exp["volatility"],
            exp["pace"],
            exp["mechanics"],
            exp["complexity"],
            exp["audiovisual"],
            exp["reward_structure"],
        ]
        vectors.append(vec)
    return np.array(vectors)


# ---------------------------------------------------------
# Scale experiences using their own scaler
# ---------------------------------------------------------
def scale_experiences(exp_vectors):
    scaler = StandardScaler()
    return scaler.fit_transform(exp_vectors)


# ---------------------------------------------------------
# Reduce experience vectors using their own PCA
# ---------------------------------------------------------
def reduce_experiences(exp_scaled):
    pca = PCA(n_components=2)
    return pca.fit_transform(exp_scaled)


# ---------------------------------------------------------
# Main recommendation function
# ---------------------------------------------------------
def recommend(player_id):
    # Load player models
    player_scaler, player_pca = load_player_models()

    # Load player vector
    player_raw = load_player(player_id)
    player_scaled = player_scaler.transform(player_raw)
    player_vec = player_pca.transform(player_scaled)  # shape (1, 2)

    # Load experiences
    experiences = load_experiences()
    exp_vectors = embed_experiences(experiences)

    # Process experiences separately
    exp_scaled = scale_experiences(exp_vectors)
    exp_vecs = reduce_experiences(exp_scaled)  # shape (N, 2)

    # Compute similarity
    sims = cosine_similarity(player_vec, exp_vecs)[0]

    # Top 3 recommendations
    top_idx = sims.argsort()[::-1][:3]

    return [(experiences[i]["name"], float(sims[i])) for i in top_idx]


# ---------------------------------------------------------
# Debug run
# ---------------------------------------------------------
if __name__ == "__main__":
    print("Top recommendations for player 0:")
    print(recommend(0))
