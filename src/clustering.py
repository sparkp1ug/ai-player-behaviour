from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

from feature_engineering import FEATURE_COLUMNS


def choose_optimal_k(X, k_range: range, out_dir: Path) -> int:
    """
    Choose the optimal number of clusters (k) using the silhouette score.

    Args:
        X: Array-like, shape (n_samples, n_features) containing the features for clustering.
        k_range: Range of k values to evaluate.
        out_dir: Directory to save the plot of the silhouette scores.

    Returns:
        The optimal number of clusters (k).
    """

    best_k = None
    best_score = -1
    silhouette_scores = []

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(X)
        score = silhouette_score(X, labels)
        silhouette_scores.append(score)

        if score > best_score:
            best_score = score
            best_k = k

    # Plot silhouette scores for each k
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(k_range, silhouette_scores, marker='o')
    ax.set_xlabel('Number of Clusters (k)')
    ax.set_ylabel('Silhouette Score')
    ax.set_title('Optimal k based on Silhouette Score')

    # Save the plot
    plt.savefig(out_dir / 'silhouette_scores.png')
    plt.close()

    return best_k

def run_clustering(features_path: Path, out_dir: Path, k: int | None, k_min: int, k_max: int) -> None:
    """
    Run the clustering process on the provided features.

    Args:
        features_path: Path to the CSV file containing the features.
        out_dir: Directory to save the output plots and results.
        k: Optional number of clusters. If None, the optimal k will be determined.
        k_min: Minimum number of clusters to consider if k is None.
        k_max: Maximum number of clusters to consider if k is None.
    """

    # Load the features from the CSV file
    df = pd.read_csv(features_path)

    X = df[FEATURE_COLUMNS].values

    # Create output directory if it doesn't exist
    out_dir.mkdir(parents=True, exist_ok=True)

    # Determine optimal k if not provided
    if k is None:
        k = choose_optimal_k(X, range(k_min, k_max + 1), out_dir)
        print(f"Optimal number of clusters (k) determined to be: {k}")

    # Fit KMeans with the chosen number of clusters
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = kmeans.fit_predict(X)
    silhouette = silhouette_score(X, labels)
    print(f"Silhouette score for k={k}: {silhouette:.4f}")

    pca = PCA(n_components=2)
    coords = pca.fit_transform(X)
    explained_variance = pca.explained_variance_ratio_
    print(f"PCA explained variance: PC1={explained_variance[0]:.2%}, PC2={explained_variance[1]:.2%}")

    # Save the cluster labels to a new CSV file
    result = df[['player_id', 'persona_id']].copy()
    result['cluster'] = labels
    result['pca_x'] = coords[:, 0]
    result['pca_y'] = coords[:, 1]
    result.to_csv(out_dir / 'clustered_data.csv', index=False)

    #Sanity check: Ensure that the number of unique clusters matches the expected k
    cross_tab = pd.crosstab(result['persona_id'], result['cluster'])
    print("\nPersona ID vs Cluster Crosstab:")
    print(cross_tab)

    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(result['pca_x'], result['pca_y'], c=result['cluster'], cmap='tab10', s=30, alpha=0.7)
    ax.set_xlabel(f'PC1 ({explained_variance[0]:.2%} variance)')
    ax.set_ylabel(f'PC2 ({explained_variance[1]:.2%} variance)')
    ax.set_title(f'Player Behavior Clusters (k={k})')
    legend = ax.legend(*scatter.legend_elements(), title='Cluster', loc='best')
    ax.add_artist(legend)
    plt.tight_layout()
    plt.savefig(out_dir / 'cluster_pca.png', dpi=150)
    plt.close(fig)

    print (f"\nResults saved to {out_dir / 'clustered_data.csv'} and {out_dir / 'cluster_pca.png'}")

def main() -> None:
    parser = argparse.ArgumentParser(description="Run clustering on player behavior features.")
    parser.add_argument("--features-path", type=Path,
                        default=Path("data/player_features_scaled.csv"),
                        help="Path to the CSV file containing features.")
    parser.add_argument("--out-dir", type=Path, default=Path("data/"),
                        help="Directory to save output plots and results.")
    parser.add_argument("--k", type=int, default=None,
                        help="Number of clusters. If omitted, the optimal k is chosen by silhouette score.")
    parser.add_argument("--k-min", type=int, default=2,
                        help="Minimum number of clusters to consider if k is not provided.")
    parser.add_argument("--k-max", type=int, default=10,
                        help="Maximum number of clusters to consider if k is not provided.")

    args = parser.parse_args()

    run_clustering(args.features_path, args.out_dir, args.k, args.k_min, args.k_max)

if __name__ == "__main__":
    main()