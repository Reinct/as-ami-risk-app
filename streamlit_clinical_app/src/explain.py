"""SHAP explanations -- the same algorithm as R's `iml::Shapley` / `permshap`.

Model-agnostic: the caller passes any object exposing
``predict_proba_death(X) -> (n,)``, ``load_background()`` and ``features``
(:class:`logistic.LogisticModel` and :class:`forest.RangerForest` both qualify),
so the same code explains the delivered logistic regression and the comparison
random forest.

Why not the `shap` package
--------------------------
`shap.TreeExplainer` only accepts scikit-learn tree objects, and would not work
for the logistic regression at all. Instead this module implements **sampled
permutation SHAP** directly -- the very algorithm behind the
`Sampling version of permutation SHAP` line in the R log. Explanations here are
therefore from the same method as the paper's SHAP beeswarm figure.

Algorithm
---------
For random permutations pi and the feature subset S(pi, j):

    phi_j = E_pi[ f(S u {j}) - f(S) ]

where f(S) takes the patient's true values for features in S and background
values for the rest, then averages over the background. Averaging over
`n_perm` permutations gives the estimate. The empty-set value is
``base_value = E[f(background)]``.

Measured (`output/_test_py_forest.py`): ``base_value + sum(phi)`` differs from
``f(x)`` by less than 2e-17, i.e. local accuracy holds to numerical precision.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

__all__ = ["permutation_shap", "explain_patient"]


def _coalition_value(
    predict_fn: Callable[[np.ndarray], np.ndarray],
    x: np.ndarray,
    background: np.ndarray,
    mask: np.ndarray,
    memo: dict,
    cache: bool,
) -> float:
    """f(S): patient values where mask is True, background elsewhere, averaged."""
    if cache:
        key = mask.tobytes()
        hit = memo.get(key)
        if hit is not None:
            return hit
    z = np.where(mask[None, :], x[None, :], background)
    v = float(predict_fn(z).mean())
    if cache:
        memo[key] = v
    return v


def permutation_shap(
    predict_fn: Callable[[np.ndarray], np.ndarray],
    x: np.ndarray,
    background: np.ndarray,
    n_perm: int = 64,
    seed: int = 2026,
    cache: bool = True,
) -> tuple[np.ndarray, float]:
    """Return (phi, base_value).

    predict_fn: (n, m) -> (n,) P(Death) prediction function
    x:          (m,) feature vector of one patient
    background: (B, m) background distribution
    n_perm:     number of sampled permutations (more is more stable, linear cost)
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    bg = np.asarray(background, dtype=np.float64)
    m = x.shape[0]
    if bg.shape[1] != m:
        raise ValueError(
            f"background has {bg.shape[1]} columns, patient has {m}"
        )

    rng = np.random.default_rng(seed)
    memo: dict = {}
    empty = np.zeros(m, dtype=bool)
    base = _coalition_value(predict_fn, x, bg, empty, memo, cache)

    phi = np.zeros(m, dtype=np.float64)
    for _ in range(int(n_perm)):
        order = rng.permutation(m)
        mask = np.zeros(m, dtype=bool)
        prev = base
        for j in order:
            mask[j] = True
            cur = _coalition_value(predict_fn, x, bg, mask, memo, cache)
            phi[j] += cur - prev
            prev = cur
    return phi / float(n_perm), base


def explain_patient(
    forest,
    x: np.ndarray,
    background: np.ndarray | None = None,
    n_perm: int = 64,
    seed: int = 2026,
    bg_size: int = 50,
) -> pd.DataFrame:
    """Tidy a single patient's SHAP result into a table sorted by |phi|.

    Columns: feature, label, value, phi, direction
    `direction` is "risk" (raises mortality risk) or "protective" (lowers it).
    """
    from .labels import feature_label, format_value  # local import avoids a cycle

    x = np.asarray(x, dtype=np.float64).ravel()
    if background is None:
        background = forest.load_background()
    bg = np.asarray(background, dtype=np.float64)[: int(bg_size)]

    phi, base = permutation_shap(
        forest.predict_proba_death, x, bg, n_perm=n_perm, seed=seed
    )
    df = pd.DataFrame({"feature": forest.features, "phi": phi})
    df["label"] = [feature_label(k) for k in df["feature"]]
    df["value"] = x
    df["value_text"] = [format_value(k, v) for k, v in zip(df["feature"], x)]
    df["direction"] = np.where(df["phi"] >= 0, "risk", "protective")
    df = df.reindex(df["phi"].abs().sort_values(ascending=False).index)
    df.attrs["base_value"] = base
    return df.reset_index(drop=True)
