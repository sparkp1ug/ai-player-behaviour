import numpy as np
import pandas as pd
import json
from sklearn.metrics.pairwise import cosine_similarity
import joblib

def load_models():
    scaler = joblib.load("models/scaler.pkl")
    pca = joblib.load("models/pca.pkl")
    return scaler, pca

def load_player(player_id):
    df = pd.read_parquet("data/players.parquet")
    return df.iloc[player_id].values.reshape(1, -1)

def load_experiences():
    with open("data/experiences.json") as f:
        return json.load(f)

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

def recommend(player_id):
    scaler, pca = load_models()
    player_raw = load_player(player_id)
    player_scaled = scaler.transform(player_raw)
    player_vec = pca.transform(player_scaled)

    experiences = load_experiences()
    exp_vectors = embed_experiences(experiences)
    exp_scaled = scaler.transform(exp_vectors)
    exp_vecs = pca.transform(exp_scaled)

    sims = cosine_similarity(player_vec, exp_vecs)[0]
    top_idx = sims.argsort()[::-1][:3]

    return [(experiences[i]["name"], sims[i]) for i in top_idx]

if __name__ == "__main__":
    print(recommend(0))
