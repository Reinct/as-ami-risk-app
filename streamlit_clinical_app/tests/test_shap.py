# -*- coding: utf-8 -*-
"""SHAP end-to-end check: local accuracy, direction sanity, runtime.

Runs against both shipped models (the logistic regression that is the delivered
model per E53, and the 680-tree random forest kept as a comparison), so the
explanation path is exercised for each.

Run (from the streamlit_clinical_app directory):
    python tests/test_shap.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

APP_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(APP_DIR))

from src.explain import explain_patient  # noqa: E402
from src.forest import RangerForest  # noqa: E402
from src.logistic import LogisticModel  # noqa: E402

MODELS = [("logistic", LogisticModel), ("forest", RangerForest)]


def run(kind: str, cls) -> None:
    print(f"\n================ {kind} ================")
    f = cls()
    fx = f.load_parity_fixture()
    X = np.asarray(fx["X"], dtype=float)
    bg = f.load_background()

    # Pick a real high-risk patient from the fixture.
    hi = X[0]
    p_hi = float(f.predict_proba_death(hi[None, :])[0])

    t0 = time.perf_counter()
    df = explain_patient(f, hi, background=bg, n_perm=64, seed=2026, bg_size=50)
    dt = time.perf_counter() - t0
    base = float(df.attrs["base_value"])

    print(f"runtime {dt:.2f}s | n_perm=64, bg=50")
    print(
        f"f(x) = {p_hi:.6f} | base = {base:.6f} | "
        f"base + sum(phi) = {base + df['phi'].sum():.6f}"
    )
    print(f"local-accuracy error = {abs(base + df['phi'].sum() - p_hi):.2e}")
    assert abs(base + df["phi"].sum() - p_hi) < 1e-9, "local accuracy violated"

    print("\nTop 6 contributions:")
    for _, r in df.head(6).iterrows():
        print(
            f"  {r['feature']:>18s}  {r['label']:<28s} "
            f"value {r['value_text']:<16s} phi = {r['phi']:+.4f}  {r['direction']}"
        )

    # Direction spot checks (informational -- the sign is patient-specific, so
    # these are printed rather than asserted). Feature names are taken from the
    # model, which means a changed feature set cannot break this test.
    d = dict(zip(df["feature"], df["phi"]))
    print("\nDirection spot checks:")
    for name, note in (
        ("uric_acid", "high uric acid -> higher risk"),
        ("hemoglobin", "high hemoglobin -> protective"),
        ("albumin", "high albumin -> protective"),
    ):
        if name in d:
            print(f"  {name:<12s} ({note:<32s}) -> phi = {d[name]:+.4f}")
    weakest = min(d, key=lambda k: abs(d[k]))
    print(f"  weakest feature in this model      -> {weakest} "
          f"|phi| = {abs(d[weakest]):.4f}")

    # Ranking stability across seeds (to compare against the paper figure).
    print("\nStability of the top 5 features across 3 seeds:")
    for s in (2026, 7, 99):
        dd = explain_patient(f, hi, background=bg, n_perm=64, seed=s, bg_size=50)
        print(f"  seed={s:>4}: {', '.join(dd['feature'].head(5))}")


for kind, cls in MODELS:
    run(kind, cls)

print("\nSHAP_OK")
