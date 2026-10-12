"""Bit-exact Python reproduction of the delivered R logistic regression.

Background
----------
Per E53 the delivered (deployed / app) model is a plain logistic regression
(R `classif.log_reg`, i.e. `stats::glm(family = binomial)` on the 14 selected
features plus an intercept). It was chosen over the 680-tree random forest
because it is better calibrated (slope 1.114 vs 1.387), has the lowest Brier
score (0.1273 vs 0.1286), a perfectly monotone seven-band risk gradient
(2.3% -> 55.2%, a 24-fold span), the best net benefit across the clinically
relevant 20-60% threshold range, and the fewest missed deaths at the
cross-validated Youden cut (12/58 vs 12/58, with dAUC = -0.0043, DeLong
p = 0.836, i.e. the two are statistically indistinguishable).

Because the model is linear, reproducing it needs no tree export: the R side
writes the intercept and the 14 coefficients to `model/logistic.json`, and R
verifies `cbind(1, X) %*% beta -> 1/(1+exp(-eta))` against mlr3 to < 1e-12
before writing.

The 680-tree random forest is still exported (`forest.json`) and remains
available as a comparison model; see `forest.RangerForest`.

Interface
---------
Deliberately duck-type identical to :class:`forest.RangerForest` so the app and
the permutation-SHAP code work with either model unchanged:
``features``, ``n_features``, ``meta``, ``youden_cut``, ``positive_label``,
``negative_label``, ``predict_proba_death(X)``, ``predict(X)``,
``median_profile()``, ``load_background()``, ``load_parity_fixture()``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

__all__ = ["LogisticModel", "MODEL_DIR"]

MODEL_DIR = Path(__file__).resolve().parent.parent / "model"


class LogisticModel:
    """Loads `logistic.json` and reproduces the R glm's predictions."""

    kind = "logistic"
    display_name = "Logistic regression"

    def __init__(self, model_dir: str | Path = MODEL_DIR):
        self.model_dir = Path(model_dir)
        self.meta: dict = json.loads(
            (self.model_dir / "meta.json").read_text(encoding="utf-8")
        )
        spec = json.loads(
            (self.model_dir / "logistic.json").read_text(encoding="utf-8")
        )
        self.spec = spec

        self.features: list[str] = list(spec["features"])
        self.n_features: int = len(self.features)
        self.positive_label: str = spec["positive_label"]
        self.negative_label: str = self.meta.get("negative_label")

        self.intercept: float = float(spec["intercept"])
        coeffs = np.asarray(spec["coefficients"], dtype=np.float64)
        if coeffs.shape[0] != self.n_features:
            raise ValueError(
                f"logistic.json has {coeffs.shape[0]} coefficients but "
                f"{self.n_features} features"
            )
        self.coefficients = coeffs
        # Full beta vector with the intercept in slot 0, for linear algebra.
        self._beta = np.concatenate([[self.intercept], coeffs])

        # Both the top level and models.logistic hold the same override.
        self.youden_cut: float = float(
            (self.meta.get("models", {}).get("logistic") or {}).get("youden_cut")
            or spec.get("youden_cut")
            or self.meta.get("youden_cut")
        )
        self.perf_metrics = (self.meta.get("models", {}).get("logistic") or {}).get(
            "perf_metrics"
        )
        self.odds_ratio_per_sd = np.asarray(
            spec.get("odds_ratio_per_sd") or [], dtype=np.float64
        )

    # ------------------------------------------------------------------ #
    # Prediction
    # ------------------------------------------------------------------ #
    def linear_predictor(self, X) -> np.ndarray:
        """Return eta = intercept + X @ beta, shape (n,)."""
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X[None, :]
        if X.shape[1] != self.n_features:
            raise ValueError(
                f"expected {self.n_features} columns, got {X.shape[1]}"
            )
        return self.intercept + X @ self.coefficients

    def predict_proba_death(self, X) -> np.ndarray:
        """Return P(Death) with shape (n,). Column order must match `self.features`."""
        eta = self.linear_predictor(X)
        # Clipped expit: identical to 1/(1+exp(-eta)) across the whole realizable
        # range (|eta| < 700 here) but cannot overflow on absurd user input.
        out = np.empty_like(eta)
        pos = eta >= 0
        out[pos] = 1.0 / (1.0 + np.exp(-eta[pos]))
        e = np.exp(eta[~pos])
        out[~pos] = e / (1.0 + e)
        return out

    def predict(self, X) -> dict:
        """Full prediction: probability, risk stratum, and threshold flag."""
        p = np.atleast_1d(self.predict_proba_death(X))
        cut = self.youden_cut
        low = cut / 2.0
        risk = np.where(p < low, "Low", np.where(p < cut, "Intermediate", "High"))
        return {
            "prob_death": p,
            "prob_alive": 1.0 - p,
            "risk_level": risk,
            "cut": cut,
            "low_cut": low,
            "over_cut": p >= cut,
        }

    # ------------------------------------------------------------------ #
    # Convenience accessors (same semantics as RangerForest)
    # ------------------------------------------------------------------ #
    def median_profile(self) -> dict:
        """Training-set medians, used as widget defaults."""
        out = {}
        for name in self.features:
            rng = self.meta["ranges"].get(name) or {}
            disc = self.meta["discrete_values"].get(name) or []
            if disc:
                out[name] = float(sorted(disc)[0])   # reference level for categoricals
            elif rng:
                out[name] = float(rng["med"])
            else:
                out[name] = 0.0
        return out

    def load_background(self) -> np.ndarray:
        """SHAP background distribution (200 randomly drawn training rows)."""
        obj = json.loads(
            (self.model_dir / "background.json").read_text(encoding="utf-8")
        )
        return np.asarray(obj["X"], dtype=np.float64)

    def load_parity_fixture(self) -> dict:
        """The 25-row fixture R left behind, for parity verification."""
        return json.loads(
            (self.model_dir / "parity_logistic.json").read_text(encoding="utf-8")
        )

    # ------------------------------------------------------------------ #
    # Reporting
    # ------------------------------------------------------------------ #
    def coefficient_table(self) -> list[dict]:
        """Human-readable coefficients with per-SD odds ratios (for the Model Info tab)."""
        rows = []
        for i, name in enumerate(self.features):
            rows.append(
                {
                    "feature": name,
                    "coefficient": float(self.coefficients[i]),
                    "odds_ratio_per_sd": (
                        float(self.odds_ratio_per_sd[i])
                        if self.odds_ratio_per_sd.shape[0] > i
                        else None
                    ),
                }
            )
        rows.sort(key=lambda r: abs(r["coefficient"]), reverse=True)
        return rows
