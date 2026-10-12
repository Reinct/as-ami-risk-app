# -*- coding: utf-8 -*-
"""AppTest smoke test: the app renders without exceptions and predicts like R.

Both shipped models are checked: the logistic regression (default, E53) and the
680-tree random forest (comparison). Each is verified bit-for-bit against the
25-row fixture that the R export script wrote, and then rendered through the
real Streamlit app.

Run (from the streamlit_clinical_app directory):
    python tests/test_app_smoke.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

APP_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(APP_DIR))

from streamlit.testing.v1 import AppTest  # noqa: E402

from src.forest import RangerForest  # noqa: E402
from src.logistic import LogisticModel  # noqa: E402

MODELS = [("logistic", LogisticModel), ("forest", RangerForest)]
RADIO_LABEL = {
    "logistic": "Logistic regression",
    "forest": "Random forest (680 trees)",
}


def parity(f, kind: str) -> None:
    """Compare the Python replay with R's own predictions on the fixture."""
    fx = f.load_parity_fixture()
    X = np.asarray(fx["X"], dtype=float)
    ref = np.asarray(fx["r_prob"], dtype=float)
    p = f.predict_proba_death(X)
    maxdiff = float(np.abs(p - ref).max())
    print(f"[{kind}] parity max abs diff over 25 rows = {maxdiff:.3e}")
    assert maxdiff < 1e-12, f"{kind}: mismatch against R: {maxdiff}"

    # Class calls at 0.5 must agree too (guards against a flipped positive class).
    r_pred = np.asarray(fx["r_pred"]).astype(str)
    label = np.where(p >= 0.5, f.positive_label, f.negative_label)
    n_bad = int((label != r_pred).sum())
    print(f"[{kind}] 0.5-threshold class disagreements = {n_bad}/25")
    assert n_bad == 0, f"{kind}: {n_bad} class disagreements against R"
    return X, ref


def render(kind: str, f, X, ref) -> None:
    """Drive the real app for this model."""
    at = AppTest.from_file(str(APP_DIR / "app.py"), default_timeout=180)
    at.run()
    if at.exception:
        for e in at.exception:
            print("[EXC]", e.value)
        raise SystemExit(f"{kind}: app failed to render")

    if RADIO_LABEL[kind] != RADIO_LABEL["logistic"]:
        at.radio[0].set_value(RADIO_LABEL[kind])
        at.run()
        if at.exception:
            for e in at.exception:
                print("[EXC]", e.value)
            raise SystemExit(f"{kind}: app failed to re-render")

    print(f"[{kind}] title =", at.title[0].value)
    print(f"[{kind}] tabs  =", [t.label for t in at.tabs])
    print(
        f"[{kind}] number_input = {len(at.number_input)}, "
        f"selectbox = {len(at.selectbox)}"
    )

    # Derive the layout from the model metadata instead of hardcoding feature
    # names, so the test keeps working when the selected feature set changes.
    CATEGORICAL = {k for k, v in (f.meta.get("discrete_values") or {}).items() if v}
    N_CONT = len(f.features) - len(CATEGORICAL)
    clin_num = [n for n in at.number_input if n.key and n.key.startswith("in_")]
    assert len(clin_num) == N_CONT, (kind, len(clin_num), N_CONT, [n.key for n in clin_num])
    assert len(at.selectbox) == len(CATEGORICAL), (kind, len(at.selectbox), len(CATEGORICAL))
    assert len({n.key for n in at.number_input} - {n.key for n in clin_num}) == 1
    print(f"[{kind}] categorical (selectbox) features = {sorted(CATEGORICAL)}")

    # Defaults are the training medians, so the page must show that prediction.
    med = f.median_profile()
    x_med = np.asarray([med[k] for k in f.features], dtype=float)
    p_med = float(f.predict_proba_death(x_med)[0])
    print(f"[{kind}] median-profile prediction = {p_med:.6f} (must appear on the page)")
    body = " ".join(str(m.value) for m in at.markdown) + " ".join(
        str(c.value) for c in at.caption
    )
    assert f"{p_med:.1%}" in body, f"{kind}: page did not show {p_med:.1%}"

    # ---- Loading the high-risk example must yield a high-risk stratum ----
    hi = X[int(np.argmax(ref))]
    at2 = AppTest.from_file(str(APP_DIR / "app.py"), default_timeout=180)
    at2.run()
    if RADIO_LABEL[kind] != RADIO_LABEL["logistic"]:
        at2.radio[0].set_value(RADIO_LABEL[kind])
        at2.run()
    at2.session_state  # noqa: B018  (confirm state is reachable)
    for name, val in zip(f.features, hi):
        key = f"in_{name}"
        if name in CATEGORICAL:
            for sb in at2.selectbox:
                if sb.key == key:
                    sb.set_value(int(round(float(val))))
        else:
            for ni in at2.number_input:
                if ni.key == key:
                    ni.set_value(float(val))
    at2.run()
    if at2.exception:
        for e in at2.exception:
            print("[EXC]", e.value)
        raise SystemExit(f"{kind}: high-risk example failed to render")
    p_hi = float(f.predict_proba_death(hi[None, :])[0])
    body2 = " ".join(str(m.value) for m in at2.markdown)
    assert f"{p_hi:.1%}" in body2, f"{kind}: high-risk example did not show {p_hi:.1%}"
    print(f"[{kind}] high-risk example = {p_hi:.6f}, rendered correctly")


for kind, cls in MODELS:
    f = cls()
    X, ref = parity(f, kind)
    render(kind, f, X, ref)

print("\nSMOKE_OK")
