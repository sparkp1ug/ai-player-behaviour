import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import joblib
from pathlib import Path

def load_data():
    return pd.read_parquet("data/players.parquet")

def preprocess(df):
    features = df.values
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)
    return scaled, scaler

def reduce_dimensionality(scaled):
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(scaled)
    return reduced, pca

def cluster(reduced):
    kmeans = KMeans(n_clusters=5, random_state=42)
    labels = kmeans.fit_predict(reduced)
    return labels, kmeans

if __name__ == "__main__":
    df = load_data()
    scaled, scaler = preprocess(df)
    reduced, pca = reduce_dimensionality(scaled)
    labels, kmeans = cluster(reduced)

    df["cluster"] = labels
    df.to_parquet("data/players_clustered.parquet")

    Path("models").mkdir(exist_ok=True)
    joblib.dump(scaler, "models/scaler.pkl")
    joblib.dump(pca, "models/pca.pkl")
    joblib.dump(kmeans, "models/kmeans.pkl")

    print("Clustering complete → data/players_clustered.parquet")
