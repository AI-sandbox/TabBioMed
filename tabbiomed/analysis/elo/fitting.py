from __future__ import annotations
from sklearn.linear_model import LogisticRegression


# Adapted from EloHelper.compute_mle_elo: https://github.com/autogluon/tabarena/blob/a81463a7f6512e3d433948269562d493f22b5054/bencheval/bencheval/elo_utils.py
def fit_coefficients(
    design,
    labels,
    weights,
    *,
    regularization=1000000.0,
    max_iter=10000,
    tolerance=1e-10,
    solver="lbfgs",
    random_state=42,
):
    model = LogisticRegression(
        fit_intercept=False,
        C=regularization,
        max_iter=max_iter,
        tol=tolerance,
        solver=solver,
        random_state=random_state,
    )
    model.fit(design, labels, sample_weight=weights)
    return model.coef_.ravel()
