"""
Behavioural embedding model.

Learns a low-dimensional embedding of each player's behaviour by training a
small autoencoder (encoder -> bottleneck -> decoder) to reconstruct the
scaled behavioural feature vector. The bottleneck activations are the
"experience embedding."

This is implemented from scratch in NumPy (full-batch gradient descent, one
hidden layer) rather than via a deep learning framework — the dataset here
is a few hundred rows and a handful of features, so a framework buys nothing
except an extra dependency.

The architecture is deliberately the smallest thing that is still genuinely
an autoencoder:

    pre1  = X @ W1 + b1          X is (n, d), W1 is (d, k)
    Z     = tanh(pre1)           the bottleneck / embedding, (n, k)
    X_hat = Z @ W2 + b2          linear decoder back to (n, d)
    L     = mean over rows of the row's squared error

with the backward pass

    d_out  = 2/n * (X_hat - X)
    dW2    = Z.T @ d_out         db2 = d_out.sum(0)
    dZ     = d_out @ W2.T
    d_pre1 = dZ * (1 - Z**2)     tanh'(pre1) = 1 - tanh(pre1)^2
    dW1    = X.T @ d_pre1        db1 = d_pre1.sum(0)

Those five lines are the whole model. `gradient_check()` verifies them
against central finite differences, so the derivation is tested rather than
asserted.

On whether this beats PCA: it does not, and the measurement is printed every
run rather than hidden. At convergence it reaches a reconstruction loss of
~4.040 against PCA's 4.020 on the same 3-d bottleneck — within half a percent
of the optimal *linear* map, from above. Read that as the result it is: these
13 standardised behavioural features are close enough to linear that a tanh
autoencoder, free to be nonlinear, converges to essentially the PCA solution
anyway. The nonlinearity buys nothing in reconstruction here.

What it does buy is a bounded embedding. tanh confines every player to
(-1, 1) on each axis, so no outlier can sit arbitrarily far out and dominate
a distance metric computed in this space — which matters if this feeds a
similarity-based recommender. That is the honest reason to prefer it, not
accuracy.

Separate from clustering.py: clustering operates directly on the scaled
features, and this embedding is what recommendation_engine.py can optionally
draw on for a denser player representation. They're two different views of
the same underlying features, not a pipeline where one feeds the other.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from feature_engineering import FEATURE_COLUMNS  # noqa: E402


@dataclass
class AutoEncoder:
    """Weights only — no optimiser state, no framework, no hidden mutation."""
    W1: np.ndarray   # (d, k) encoder
    b1: np.ndarray   # (k,)
    W2: np.ndarray   # (k, d) decoder
    b2: np.ndarray   # (d,)

    @property
    def n_features(self) -> int:
        return self.W1.shape[0]

    @property
    def n_components(self) -> int:
        return self.W1.shape[1]

    def encode(self, X: np.ndarray) -> np.ndarray:
        """The embedding. Bounded to (-1, 1) by tanh, which is a real
        difference from PCA scores: no player can sit arbitrarily far out on
        a component, so one extreme player cannot dominate a distance metric
        computed in this space."""
        return np.tanh(X @ self.W1 + self.b1)

    def decode(self, Z: np.ndarray) -> np.ndarray:
        return Z @ self.W2 + self.b2

    def reconstruct(self, X: np.ndarray) -> np.ndarray:
        return self.decode(self.encode(X))


def reconstruction_loss(X: np.ndarray, X_hat: np.ndarray) -> float:
    """Mean over rows of each row's summed squared error.

    Summing over features and averaging over rows (rather than averaging over
    both) is what makes the gradient exactly 2/n * (X_hat - X) — matching the
    derivation in the module docstring, with n the number of rows.
    """
    return float(np.mean(np.sum((X_hat - X) ** 2, axis=1)))


def init_weights(n_features: int, n_components: int, rng: np.random.Generator) -> AutoEncoder:
    """Xavier/Glorot scaling: variance 1/fan_in keeps pre-activations in
    tanh's responsive range instead of saturating it flat at +/-1, where the
    (1 - Z**2) term would kill the gradient before training starts."""
    return AutoEncoder(
        W1=rng.normal(0.0, np.sqrt(1.0 / n_features), size=(n_features, n_components)),
        b1=np.zeros(n_components),
        W2=rng.normal(0.0, np.sqrt(1.0 / n_components), size=(n_components, n_features)),
        b2=np.zeros(n_features),
    )


def gradients(model: AutoEncoder, X: np.ndarray) -> tuple[dict[str, np.ndarray], float]:
    """One forward and one backward pass over the whole batch.

    Returns the gradient of `reconstruction_loss` with respect to each
    parameter, plus the loss itself.
    """
    n = X.shape[0]

    Z = np.tanh(X @ model.W1 + model.b1)      # (n, k)
    X_hat = Z @ model.W2 + model.b2           # (n, d)

    d_out = 2.0 / n * (X_hat - X)             # (n, d)
    dW2 = Z.T @ d_out                         # (k, d)
    db2 = d_out.sum(axis=0)                   # (d,)

    dZ = d_out @ model.W2.T                   # (n, k)
    d_pre1 = dZ * (1.0 - Z ** 2)              # tanh'
    dW1 = X.T @ d_pre1                        # (d, k)
    db1 = d_pre1.sum(axis=0)                  # (k,)

    grads = {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}
    return grads, reconstruction_loss(X, X_hat)


def train(
    X: np.ndarray,
    n_components: int = 3,
    epochs: int = 12000,
    learning_rate: float = 0.01,
    seed: int = 42,
    verbose: bool = False,
) -> tuple[AutoEncoder, list[float]]:
    """Full-batch gradient descent. No mini-batches, no momentum, no Adam —
    with a few hundred rows the full batch fits trivially and plain GD is one
    line, which keeps the whole model defensible end to end.

    Defaults chosen by sweep: lr=0.05 leaves the loss oscillating with a
    range of ~0.14 over the last 200 epochs, lr=0.01 over 12k epochs settles
    to ~0.0003. Slow and converged beats fast and bouncing when the whole run
    takes under a second.
    """
    rng = np.random.default_rng(seed)
    model = init_weights(X.shape[1], n_components, rng)
    history: list[float] = []

    for epoch in range(epochs):
        grads, loss = gradients(model, X)
        model.W1 -= learning_rate * grads["W1"]
        model.b1 -= learning_rate * grads["b1"]
        model.W2 -= learning_rate * grads["W2"]
        model.b2 -= learning_rate * grads["b2"]
        history.append(loss)
        if verbose and epoch % max(epochs // 10, 1) == 0:
            print(f"  epoch {epoch:>5}  loss {loss:.5f}")

    history.append(reconstruction_loss(X, model.reconstruct(X)))
    return model, history


def gradient_check(seed: int = 0, tol: float = 1e-5) -> float:
    """Verify the analytic backward pass against central finite differences.

    Checks a sample of entries in every parameter and returns the worst
    relative error. This is the test that the five-line derivation in the
    module docstring is actually correct — the cheapest possible insurance
    against a transposed matrix that still runs and still reduces the loss.

    Tolerance is 1e-5, the conventional choice, not something tighter.
    Central differences at eps=1e-6 carry their own truncation and
    cancellation error of order 1e-7 in f64: seeds sampled here land between
    3e-09 and 1.4e-07 on correct code. A genuine backprop error is O(1)
    wrong, not O(1e-7), so the two orders of magnitude of slack cost nothing
    in detection power and buy a test that does not flake on the draw.
    """
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(12, 5))
    model = init_weights(5, 3, rng)
    analytic, _ = gradients(model, X)

    eps = 1e-6
    worst = 0.0
    for name in ("W1", "b1", "W2", "b2"):
        param = getattr(model, name)
        flat = param.ravel()
        idxs = rng.choice(flat.size, size=min(6, flat.size), replace=False)
        for i in idxs:
            original = flat[i]

            flat[i] = original + eps
            plus = reconstruction_loss(X, model.reconstruct(X))
            flat[i] = original - eps
            minus = reconstruction_loss(X, model.reconstruct(X))
            flat[i] = original

            numeric = (plus - minus) / (2 * eps)
            exact = analytic[name].ravel()[i]
            denom = max(abs(numeric), abs(exact), 1e-12)
            worst = max(worst, abs(numeric - exact) / denom)

    if worst > tol:
        raise AssertionError(f"gradient check failed: worst relative error {worst:.2e}")
    return worst


def pca_baseline(X: np.ndarray, n_components: int) -> float:
    """Reconstruction loss of a linear PCA with the same bottleneck width.

    PCA minimises exactly this quantity over all rank-k linear maps, so it is
    the honest yardstick: if the autoencoder does not beat it, the nonlinearity
    is not buying reconstruction accuracy on this data, and the docstring says
    so rather than the other way round.
    """
    from sklearn.decomposition import PCA

    pca = PCA(n_components=n_components)
    return reconstruction_loss(X, pca.inverse_transform(pca.fit_transform(X)))


def run(features_path: Path, out_path: Path, n_components: int,
        epochs: int, learning_rate: float, seed: int) -> None:
    features = pd.read_csv(features_path)
    X = features[FEATURE_COLUMNS].to_numpy(dtype=float)

    worst = gradient_check()
    print(f"gradient check passed (worst relative error {worst:.2e})")

    model, history = train(X, n_components, epochs, learning_rate, seed, verbose=True)

    baseline = reconstruction_loss(X, np.tile(X.mean(axis=0), (X.shape[0], 1)))
    pca_loss = pca_baseline(X, n_components)
    print(f"\n{X.shape[0]} players x {X.shape[1]} features -> {n_components}-d embedding")
    print(f"  predict-the-mean loss  {baseline:.4f}   (learning nothing)")
    print(f"  PCA({n_components}) loss           {pca_loss:.4f}   (optimal linear)")
    print(f"  autoencoder loss       {history[-1]:.4f}   "
          f"({'beats' if history[-1] < pca_loss else 'does not beat'} PCA)")

    embedding = model.encode(X)
    out = features[["player_id", "persona_id"]].copy()
    for i in range(n_components):
        out[f"emb_{i}"] = embedding[:, i]
    out.to_csv(out_path, index=False)
    print(f"\nwrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--features-path", type=Path,
                        default=Path("data/player_features_scaled.csv"))
    parser.add_argument("--out-path", type=Path, default=Path("data/player_embeddings.csv"))
    parser.add_argument("--n-components", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=12000)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run(args.features_path, args.out_path, args.n_components,
        args.epochs, args.learning_rate, args.seed)


if __name__ == "__main__":
    main()
