"""Bit-exact Python reproduction of the delivered R `ranger` forest.

Background
----------
The delivered model is a 680-tree probability forest trained with R `ranger`
(the `randomForest` learner). So that Streamlit Community Cloud can serve
**identical** predictions **without any R runtime**, `output/_export_model.R`
dumps every tree to JSON (`model/forest.json`) via `ranger::treeInfo()`.
That export script self-checks by replaying all 1,052 training rows through all
680 exported trees and requiring the maximum absolute deviation from ranger's
own per-tree predictions to be < 1e-12 -- measured value is exactly 0.

Why not retrain with scikit-learn
---------------------------------
ranger and scikit-learn differ in bootstrap sampling, split search, and tie
handling; retraining would be a *different* model. Here the trained trees are
replayed as-is, so predictions match the published pipeline bit for bit.

Traversal rules, verified to match ranger (`output/_diag_treeinfo.R`)
--------------------------------------------------------------------
* Node ids are 0-based, root = 0
* Exported leaves carry ``left = right = -1`` (the NA in treeInfo is rewritten to
  -1, removing the ambiguity where child pointer 0 means both "no child" and
  "node id 0", i.e. the root)
* ``feature`` is a 0-based column index, and is -1 at leaves
* Rule: ``x[feature] <= threshold`` goes left, otherwise right
* A leaf's ``prob`` is that leaf's P(Death)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

__all__ = ["RangerForest", "MODEL_DIR"]

MODEL_DIR = Path(__file__).resolve().parent.parent / "model"


class RangerForest:
    """Loads `forest.json` and reproduces ranger's probability predictions."""

    kind = "forest"
    display_name = "Random forest"

    def __init__(self, model_dir: str | Path = MODEL_DIR):
        self.model_dir = Path(model_dir)
        self.meta: dict = json.loads(
            (self.model_dir / "meta.json").read_text(encoding="utf-8")
        )
        self.features: list[str] = list(self.meta["features"])
        self.n_features: int = len(self.features)
        self.youden_cut: float = float(
            (self.meta.get("models", {}).get("forest") or {}).get("youden_cut")
            or self.meta.get("youden_cut")
        )
        self.positive_label: str = self.meta["positive_label"]
        self.negative_label: str = self.meta.get("negative_label")
        self.perf_metrics = (self.meta.get("models", {}).get("forest") or {}).get(
            "perf_metrics"
        )

        trees = json.loads(
            (self.model_dir / "forest.json").read_text(encoding="utf-8")
        )
        self.n_trees: int = len(trees)

        # Concatenate every tree into flat arrays so that predictions advance
        # every tree's cursor at once, instead of looping over trees in Python.
        offsets = np.zeros(self.n_trees + 1, dtype=np.int64)
        for i, t in enumerate(trees):
            offsets[i + 1] = offsets[i] + len(t["left"])
        total = int(offsets[-1])
        self._offset = offsets

        left = np.empty(total, dtype=np.int32)
        right = np.empty(total, dtype=np.int32)
        feature = np.empty(total, dtype=np.int32)
        threshold = np.empty(total, dtype=np.float64)
        prob = np.empty(total, dtype=np.float64)
        counts = np.empty(total, dtype=np.int64)
        for i, t in enumerate(trees):
            a, b = int(offsets[i]), int(offsets[i + 1])
            left[a:b] = t["left"]
            right[a:b] = t["right"]
            feature[a:b] = t["feature"]
            threshold[a:b] = t["threshold"]
            prob[a:b] = t["prob"]
            counts[a:b] = t["count"]
        self._left = left
        self._right = right
        self._feature = feature
        self._threshold = threshold
        self._prob = prob
        self._count = counts
        del trees  # do not keep every tree as a separate sub-dict resident

    # ------------------------------------------------------------------ #
    # Prediction
    # ------------------------------------------------------------------ #
    def predict_proba_death(self, X) -> np.ndarray:
        """Return P(Death) with shape (n,). Column order must match `self.features`."""
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X[None, :]
        if X.shape[1] != self.n_features:
            raise ValueError(
                f"expected {self.n_features} columns, got {X.shape[1]}"
            )

        n = X.shape[0]
        n_tree = self.n_trees
        base = self._offset[:-1]                     # (n_tree,)

        # node[r, t] = local node id of row r in tree t
        node = np.zeros((n, n_tree), dtype=np.int64)
        idx = base[None, :] + node                   # global index, (n, n_tree)

        # Trees are never deeper than ~100 levels; feature >= 0 means "still internal"
        for _ in range(400):
            f = self._feature[idx]
            active = f >= 0
            if not active.any():
                break
            rows, cols = np.nonzero(active)
            cur = idx[rows, cols]
            go_left = X[rows, f[rows, cols]] <= self._threshold[cur]
            nxt = np.where(go_left, self._left[cur], self._right[cur])
            node[rows, cols] = nxt
            idx[rows, cols] = base[cols] + nxt

        leaf = base[None, :] + node
        return self._prob[leaf].mean(axis=1)

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
    # Convenience accessors
    # ------------------------------------------------------------------ #
    def median_profile(self) -> dict:
        """Training-set medians, used as widget defaults."""
        out = {}
        for name in self.features:
            rng = self.meta["ranges"].get(name) or {}
            disc = self.meta["discrete_values"].get(name) or {}
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
            (self.model_dir / "parity_forest.json").read_text(encoding="utf-8")
        )
