"""The autoencoder's backward pass, and the claims its docstring makes."""

from __future__ import annotations

import numpy as np
import pytest

from embedding_model import (
    gradient_check,
    init_weights,
    pca_baseline,
    reconstruction_loss,
    train,
)


def test_gradient_check_passes():
    """The five-line backward pass against central finite differences.

    This is the test the whole module rests on: a transposed matrix in
    backprop still runs and still reduces the loss, so 'training works' is
    not evidence the derivation is right.
    """
    assert gradient_check(seed=0) < 1e-5


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_gradient_check_is_not_seed_lucky(seed):
    assert gradient_check(seed=seed) < 1e-5


def test_training_descends_monotonically():
    """Full-batch GD at a sane learning rate should never step uphill.

    Asserting monotonicity rather than "the tail is flat": on 8-dimensional
    Gaussian noise the loss is still descending at 2000 epochs, which is fine
    and not what this test is about. An increase would mean the step size is
    past the stability threshold — that is the failure worth catching.
    """
    rng = np.random.default_rng(0)
    X = rng.normal(size=(120, 8))
    _, history = train(X, n_components=3, epochs=2000, learning_rate=0.01, seed=1)

    assert history[-1] < history[0] / 2
    steps = np.diff(np.array(history))
    assert steps.max() <= 1e-9, f"loss increased by {steps.max():.2e} on some step"


def test_embedding_is_bounded_where_pca_is_not():
    """The actual argument for this model over PCA.

    tanh confines every coordinate to [-1, 1] whatever the input scale — it
    saturates to exactly +/-1 in floating point on extreme input, which is
    the bound holding, not failing. A PCA score has no such ceiling, so one
    outlier can dominate any distance computed in that space. The contrast is
    the point, so both are measured here.
    """
    from sklearn.decomposition import PCA

    rng = np.random.default_rng(0)
    X = rng.normal(size=(100, 8))
    X[0] *= 500                             # one extreme player

    model, _ = train(X, n_components=3, epochs=500, learning_rate=0.01, seed=1)
    Z = model.encode(X)
    assert np.all(np.abs(Z) <= 1.0)

    pca_scores = PCA(n_components=3).fit_transform(X)
    assert np.abs(pca_scores).max() > 100, "outlier should blow out the PCA scores"


def test_does_not_beat_pca_on_reconstruction():
    """Asserting the honest result rather than the flattering one.

    PCA is the optimal rank-k *linear* reconstruction. The docstring claims
    the autoencoder converges to essentially that solution from above on this
    kind of data rather than beating it — so that is what is tested. If a
    future change ever did beat PCA, this test failing is the right outcome:
    the docstring would need updating.
    """
    rng = np.random.default_rng(0)
    X = rng.normal(size=(150, 10))
    X = (X - X.mean(axis=0)) / X.std(axis=0)

    _, history = train(X, n_components=3, epochs=4000, learning_rate=0.01, seed=1)
    pca = pca_baseline(X, 3)

    assert history[-1] >= pca
    assert history[-1] < pca * 1.15, "should still land close to the linear optimum"


def test_loss_matches_the_gradient_convention():
    """reconstruction_loss must be the mean over rows of each row's summed
    squared error — that convention is what makes d_out exactly 2/n (X_hat-X)."""
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    X_hat = np.array([[2.0, 2.0], [3.0, 6.0]])
    assert reconstruction_loss(X, X_hat) == pytest.approx((1.0 + 4.0) / 2)


def test_weights_have_the_expected_shapes():
    model = init_weights(13, 3, np.random.default_rng(0))
    assert model.W1.shape == (13, 3) and model.b1.shape == (3,)
    assert model.W2.shape == (3, 13) and model.b2.shape == (13,)
    assert model.n_features == 13 and model.n_components == 3
