# -*- coding: utf-8 -*-
"""AS+AMI Mortality Risk Prediction Model -- Streamlit entry point.

Run locally:
    cd streamlit_clinical_app
    streamlit run app.py

Deploying to Streamlit Community Cloud: see README.md in this directory.

Two models ship with the app and can be switched from the sidebar:

* **Logistic regression** (default, E53) -- the delivered model. Chosen for
  calibration (slope 1.114), lowest Brier score, a monotone seven-band risk
  gradient, and the best net benefit over the clinically relevant threshold
  range, at no detectable cost in discrimination (dAUC = -0.0043, DeLong
  p = 0.836 against the forest).
* **Random forest** (680 trees) -- retained as a comparison model.

Both are pure-Python replays of the R pipeline; neither is retrained here.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.explain import explain_patient  # noqa: E402
from src.forest import RangerForest  # noqa: E402
from src.labels import (  # noqa: E402
    discrete_for,
    feature_label,
    feature_plain,
    format_value,
    groups_for,
    options_for,
)
from src.logistic import LogisticModel  # noqa: E402

COLOR_RISK = "#D55E00"
COLOR_PROT = "#0072B2"
RISK_STYLE = {
    "Low": ("#1B7F4B", "#E8F4EC"),
    "Intermediate": ("#B37A00", "#FFF3D6"),
    "High": ("#B3350B", "#FBE3DC"),
}

# kind -> (factory, sidebar label)
MODEL_REGISTRY = {
    "logistic": (LogisticModel, "Logistic regression"),
    "forest": (RangerForest, "Random forest (680 trees)"),
}


def _cohort(f) -> str:
    """Development-cohort description, taken from the exported metadata."""
    m = f.meta
    n, deaths = m.get("n_train"), m.get("n_events")
    if n and deaths:
        return f"{n:,} patients, {deaths} deaths"
    if n:
        return f"{n:,} patients"
    return "the development cohort"


def _holdout(f) -> str:
    """Holdout-set description, taken from the exported metadata when present."""
    h = (f.meta or {}).get("holdout") or {}
    n, deaths = h.get("n"), h.get("n_events")
    if n and deaths:
        return f"n = {n:,}, {deaths} deaths"
    if n:
        return f"n = {n:,}"
    return "the internal holdout set"


def _metric(f, name: str, column: str = "Youden threshold"):
    """Look up one number from the model's exported performance table."""
    pm = getattr(f, "perf_metrics", None)
    if not pm:
        return None
    try:
        df = pd.DataFrame(pm)
        row = df.loc[df.iloc[:, 0].astype(str).str.strip() == name]
        if row.empty:
            return None
        return float(row.iloc[0][column])
    except Exception:  # pragma: no cover - display only
        return None


def _parity_maxdiff(f) -> str:
    """Replay the R reference fixture and report the worst disagreement.

    Computed live rather than hard-coded, so the figure shown in the UI can
    never drift away from the shipped model.
    """
    try:
        fx = f.load_parity_fixture()
        x = np.asarray(fx["X"], dtype=float)
        ref = np.asarray(fx["r_prob"], dtype=float).ravel()
        got = f.predict_proba_death(x)
        if ref.shape != got.shape:
            return "n/a"
        return f"{float(np.max(np.abs(got - ref))):.3e}"
    except Exception:  # pragma: no cover - display only
        return "n/a"


st.set_page_config(
    page_title="AS+AMI Mortality Risk Prediction Model",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)


# --------------------------------------------------------------------------- #
# Cached resources (avoid re-reading JSON on every interaction)
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Loading model…")
def get_model(kind: str = "logistic"):
    factory = MODEL_REGISTRY.get(kind, (LogisticModel, ""))[0]
    return factory()


@st.cache_data(show_spinner=False)
def shap_cached(kind: str, x: tuple, n_perm: int, seed: int, bg_size: int):
    f = get_model(kind)
    df = explain_patient(
        f, np.asarray(x, dtype=float), n_perm=n_perm, seed=seed, bg_size=bg_size
    )
    base = float(df.attrs.get("base_value", np.nan))
    df.attrs.pop("base_value", None)
    return df, base


# --------------------------------------------------------------------------- #
# Input widgets
# --------------------------------------------------------------------------- #
def _step_for(name: str, hi: float) -> float:
    if name == "age":
        return 1.0
    if hi > 100:
        return 1.0
    if hi < 10:
        return 0.01
    return 0.1


def patient_inputs(f) -> dict:
    """Render the input groups; return {feature: value}.

    Only the active model's features are rendered, because the two models were
    selected to share the same 14-feature panel; if they ever diverge the
    widgets follow whichever model is currently active.
    """
    med = f.median_profile()
    ranges = f.meta["ranges"]
    levels = f.meta.get("discrete_values") or {}
    discrete = set(discrete_for(f.features, levels))
    values: dict[str, float] = {}

    cols = st.columns(3)
    for col, (title, names) in zip(cols, groups_for(f.features)):
        with col:
            st.markdown(f"##### {title}")
            for name in names:
                key = f"in_{name}"
                label = feature_label(name)
                if name in discrete:
                    opts = options_for(name, levels.get(name))
                    choices = [v for _, v in opts]
                    lookup = {v: t for t, v in opts}
                    default = int(round(med.get(name, choices[0])))
                    values[name] = st.selectbox(
                        label,
                        options=choices,
                        index=choices.index(default) if default in choices else 0,
                        format_func=lambda v, lk=lookup: lk[v],
                        key=key,
                    )
                else:
                    r = ranges.get(name) or {}
                    lo = float(r.get("min", 0.0))
                    hi = float(r.get("max", 1.0))
                    values[name] = st.number_input(
                        label,
                        min_value=0.0,
                        max_value=None,
                        value=float(med.get(name, lo)),
                        step=_step_for(name, hi),
                        format="%.4g",
                        key=key,
                        help=(
                            f"Training range {lo:g} – {hi:g}"
                            f", median {r.get('med', float('nan')):g}"
                        ),
                    )
    return values


def vector_from(values: dict, f) -> np.ndarray:
    return np.asarray([float(values[k]) for k in f.features], dtype=float)


def _risk_of(prob: float, cut: float) -> str:
    if prob < cut / 2.0:
        return "Low"
    if prob < cut:
        return "Intermediate"
    return "High"


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def risk_gauge(prob: float, cut: float) -> go.Figure:
    lo = cut / 2.0
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=prob * 100.0,
            number={"suffix": "%", "font": {"size": 46, "color": "#1A1A1A"}},
            gauge={
                "axis": {
                    "range": [0, 100],
                    "ticksuffix": "%",
                    "tickfont": {"size": 12},
                },
                "bar": {
                    "color": RISK_STYLE[_risk_of(prob, cut)][0],
                    "thickness": 0.35,
                },
                "steps": [
                    {"range": [0, lo * 100], "color": "#E8F4EC"},
                    {"range": [lo * 100, cut * 100], "color": "#FFF3D6"},
                    {"range": [cut * 100, 100], "color": "#FBE3DC"},
                ],
                "threshold": {
                    "line": {"color": "#4D4D4D", "width": 4},
                    "thickness": 0.85,
                    "value": cut * 100,
                },
            },
        )
    )
    fig.update_layout(
        height=300,
        margin=dict(l=30, r=30, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def shap_bar(df: pd.DataFrame, base: float, prob: float) -> go.Figure:
    plot = df.iloc[::-1]  # largest |phi| at the top
    colors = [COLOR_RISK if v >= 0 else COLOR_PROT for v in plot["phi"]]
    fig = go.Figure(
        go.Bar(
            x=plot["phi"],
            y=plot["label"],
            orientation="h",
            marker=dict(color=colors, line=dict(width=0)),
            customdata=np.stack([plot["value_text"], plot["phi"]], axis=-1),
            hovertemplate=(
                "<b>%{y}</b><br>Patient value: %{customdata[0]}"
                "<br>SHAP value: %{customdata[1]:+.4f}<extra></extra>"
            ),
        )
    )
    fig.add_vline(x=0, line_width=1, line_color="#9E9E9E")
    fig.update_layout(
        height=max(380, 30 * len(plot) + 120),
        margin=dict(l=10, r=20, t=46, b=40),
        xaxis_title="SHAP value (contribution to P(death))",
        yaxis_title=None,
        bargap=0.28,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        title=dict(
            text=(
                f"Baseline risk {base:.1%} → this patient {prob:.1%}"
                f"  (red = raises risk, blue = lowers risk)"
            ),
            font=dict(size=13),
            x=0.0,
        ),
    )
    fig.update_xaxes(zeroline=False, gridcolor="#E6E6E6")
    fig.update_yaxes(gridcolor="rgba(0,0,0,0)", tickfont=dict(size=12))
    return fig


# --------------------------------------------------------------------------- #
# Pages
# --------------------------------------------------------------------------- #
def render_prediction(f, values: dict) -> None:
    x = vector_from(values, f)
    res = f.predict(x)
    prob = float(res["prob_death"][0])
    cut = f.youden_cut
    level = _risk_of(prob, cut)
    fg, bg = RISK_STYLE[level]

    st.session_state["patient_x"] = tuple(x.tolist())
    st.session_state["patient_prob"] = prob

    left, right = st.columns([1.05, 1.0], gap="large")
    with left:
        st.plotly_chart(
            risk_gauge(prob, cut),
            use_container_width=True,
            config={"displayModeBar": False},
        )
    with right:
        st.markdown(
            f"""<div style="background:{bg};border-left:6px solid {fg};
            padding:14px 18px;border-radius:6px;margin-bottom:12px">
            <div style="font-size:13px;color:#555">Predicted probability of death</div>
            <div style="font-size:38px;font-weight:700;color:{fg};line-height:1.15">
            {prob:.1%}</div>
            <div style="font-size:16px;font-weight:600;color:{fg}">
            Risk stratum: {level}</div></div>""",
            unsafe_allow_html=True,
        )
        sens = _metric(f, "Sensitivity")
        spec = _metric(f, "Specificity")
        perf_line = ""
        if sens is not None and spec is not None:
            perf_line = (
                f", corresponding to sensitivity {sens:.3f} / specificity "
                f"{spec:.3f} on the internal holdout set"
            )
        st.markdown(
            f"""**Decision threshold** (Youden-optimal cut-point selected by
            cross-validation on the training set)

            - Predicted probability **≥ {cut:.4f}** → classified as high risk
            - Strata: Low &lt; {cut / 2:.4f} | Intermediate {cut / 2:.4f} – {cut:.4f}
              | High ≥ {cut:.4f}
            - This patient: **{'above' if prob >= cut else 'below'} the threshold**{perf_line}
            """
        )

    st.divider()
    st.markdown(f"##### Model performance on the internal holdout set ({_holdout(f)})")
    st.dataframe(
        pd.DataFrame(f.perf_metrics),
        hide_index=True,
        use_container_width=True,
    )
    st.caption(
        "AUC is threshold-independent. The threshold-0.5 column is shown for "
        "completeness only and is clinically uninformative for an event rate near "
        "18%; use the Youden-threshold column. At a threshold of 0.5 the likely "
        "majority-class accuracy is about 0.82, so accuracy alone must never be "
        "read as evidence of usefulness."
    )


def render_explanation(f, kind: str) -> None:
    if "patient_x" not in st.session_state:
        st.info("Enter patient data on the **Prediction** tab first.")
        return

    x = np.asarray(st.session_state["patient_x"], dtype=float)
    prob = float(st.session_state["patient_prob"])

    c1, c2, c3 = st.columns([1.1, 1.0, 1.0])
    with c1:
        n_perm = st.select_slider(
            "Sampled permutations (accuracy vs speed)",
            options=[24, 48, 64, 128, 256],
            value=64,
            help="More permutations give more stable Shapley values; "
                 "runtime grows roughly linearly.",
        )
    with c2:
        bg_size = st.select_slider(
            "Background samples", options=[30, 50, 100, 200], value=50
        )
    with c3:
        seed = st.number_input("Random seed", min_value=1, value=2026, step=1)

    st.caption(
        f"Explained model: {f.display_name}. Current patient: "
        + "; ".join(
            f"{feature_plain(k)} = {format_value(k, v)}"
            for k, v in zip(f.features, x)
        )
    )

    with st.spinner(
        f"Computing SHAP ({n_perm} permutations × {bg_size} background samples)…"
    ):
        df, base = shap_cached(
            kind, tuple(x.tolist()), int(n_perm), int(seed), int(bg_size)
        )

    st.plotly_chart(
        shap_bar(df, base, prob),
        use_container_width=True,
        config={"displayModeBar": False},
    )

    st.markdown(
        f"**Baseline risk** (mean prediction over the background "
        f"distribution) **= {base:.1%}**; **this patient = {prob:.1%}**; "
        f"sum of feature contributions = {df['phi'].sum():+.4f}, so "
        f"baseline + Σφ = prediction (local accuracy)."
    )

    show = df[["label", "value_text", "phi", "direction"]].copy()
    show.columns = ["Feature", "Patient value", "SHAP value", "Effect"]
    show["Effect"] = np.where(
        show["SHAP value"] >= 0, "Raises mortality risk", "Lowers mortality risk"
    )
    st.dataframe(
        show.style.format({"SHAP value": "{:+.4f}"}),
        hide_index=True,
        use_container_width=True,
    )
    st.caption(
        "Method: sampled permutation SHAP -- the same algorithm as `iml::permshap` "
        "used on the R side of the modelling pipeline. The paper's SHAP beeswarm "
        "figure was computed on the random forest, so for a forest explanation "
        "switch the sidebar model selector to **Random forest**."
    )


def render_model_info(f, kind: str) -> None:
    m = f.meta
    models = m.get("models") or {}
    src = (models.get(kind) or {}).get("source") or m.get("source") or {}

    st.markdown(f"##### {f.display_name}")
    st.caption(
        "The delivered model (E53) is the logistic regression; the random forest "
        "is retained as a comparison model. The two are statistically "
        "indistinguishable on discrimination (ΔAUC = −0.0043, DeLong p = 0.836), "
        "and the logistic regression is better calibrated and gives a monotone "
        "risk gradient, which is why it is the default."
        if kind == "logistic"
        else
        "Comparison model: a 680-tree probability forest (ranger). Discrimination "
        "is statistically indistinguishable from the logistic regression "
        "(ΔAUC = −0.0043, DeLong p = 0.836), but it is less well calibrated."
    )

    c1, c2, c3 = st.columns(3)
    if kind == "logistic":
        c1.metric("Parameters", f"{f.n_features + 1}")
    else:
        c1.metric("Trees", f"{getattr(f, 'n_trees', '?')}")
    c2.metric("Features", f"{f.n_features}")
    c3.metric("Training samples", f"{m['n_train']:,}")

    st.markdown("##### Model provenance")
    if kind == "logistic":
        st.markdown(
            f"""
- Learner: `{src.get('learner', 'logistic')}` (R `mlr3`, engine
  `{src.get('engine', 'stats::glm')}`, family `{src.get('family', 'binomial')}`)
- Modelling run: {src.get('run')}
- The Python side does **not refit**. `output/_export_model.R` writes the
  intercept and the {f.n_features} coefficients to `model/logistic.json`, after
  verifying `1 / (1 + exp(-(intercept + X β)))` against mlr3's own prediction
  to < 1e-12. Python evaluates the same linear predictor.
- On the 25-row reference fixture the maximum absolute deviation from R is
  **{_parity_maxdiff(f)}**.
            """
        )
        rows = f.coefficient_table()
        st.markdown("##### Coefficients (sorted by absolute magnitude)")
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Feature": feature_label(r["feature"]),
                        "Coefficient (log-odds)": r["coefficient"],
                        "Odds ratio per 1 SD": r["odds_ratio_per_sd"],
                    }
                    for r in rows
                ]
            ).style.format(
                {"Coefficient (log-odds)": "{:+.4f}", "Odds ratio per 1 SD": "{:.3f}"}
            ),
            hide_index=True,
            use_container_width=True,
        )
        st.caption(
            "Odds ratios are per one standard deviation of the training "
            "distribution, so they are comparable across features measured on "
            "different scales."
        )
    else:
        st.markdown(
            f"""
- Learner: `{src.get('learner', 'randomForest')}` (R `mlr3` with the `ranger` engine)
- Hyperparameters: `num.trees = {src.get('num_trees')}`,
  `mtry = {src.get('mtry')}`, `min.node.size = {src.get('min_node_size', 10)}`,
  `replace = {src.get('replace')}`, `sample.fraction = {src.get('sample_fraction')}`
- ranger version: {src.get('ranger_version')}; modelling run: {src.get('run')}
- The Python side does **not retrain**. `output/_export_model.R` dumps all
  {getattr(f, 'n_trees', '?')} trees verbatim to `model/forest.json` via
  `ranger::treeInfo()`; Python replays the identical split rule
  (`x[feature] <= threshold` goes left).
- On the 25-row reference fixture the maximum absolute deviation from R is
  **{_parity_maxdiff(f)}**.

> Note: `ranger` outputs raw double-precision split points, and two values that
> differ by one ulp can fall on opposite sides of a split. The export therefore
> uses 17 significant digits; with fewer, `12.219999999999998863` and
> `12.220000000000000639` both serialise to `12.22` and a handful of patients
> silently flip branches.
            """
        )

    st.markdown(f"##### The {f.n_features} model features")
    rows = []
    discrete = set(discrete_for(f.features, m.get("discrete_values") or {}))
    for name in f.features:
        r = m["ranges"].get(name) or {}
        is_disc = name in discrete
        vals = m["discrete_values"].get(name) or []
        rows.append(
            {
                "Feature": feature_label(name),
                "Type": "Categorical" if is_disc else "Continuous",
                "Training median": (
                    format_value(name, vals[0]) if is_disc
                    else f"{r.get('med', float('nan')):g}"
                ),
                "Range": (
                    " / ".join(format_value(name, v) for v in vals) if is_disc
                    else f"{r.get('min', float('nan')):g} – {r.get('max', float('nan')):g}"
                ),
            }
        )
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    st.markdown("##### Performance metrics")
    st.dataframe(
        pd.DataFrame(f.perf_metrics), hide_index=True, use_container_width=True
    )
    st.caption(
        "Both models were evaluated on the same internal holdout set "
        f"({_holdout(f)}). AUC is threshold-independent; the threshold-0.5 column "
        "is clinically uninformative at an event rate near 18%."
    )

    st.warning(
        "**For research use only.** This demonstrator was developed on a "
        f"single-centre retrospective cohort ({_cohort(f)}) and has not been "
        "externally validated. It does **not** constitute medical advice and must "
        "not be used for clinical decision-making."
    )


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    default_kind = "logistic"
    try:
        probe = get_model("logistic")
        default_kind = (probe.meta or {}).get("default_model") or default_kind
    except Exception:  # noqa: BLE001
        default_kind = "forest"
    if default_kind not in MODEL_REGISTRY:
        default_kind = "forest"

    with st.sidebar:
        st.markdown("### 🫀 AS+AMI Mortality Risk")
        st.caption(
            "In-hospital mortality risk in patients with aortic stenosis "
            "complicated by acute myocardial infarction"
        )
        st.divider()
        labels = [MODEL_REGISTRY[k][1] for k in MODEL_REGISTRY]
        kinds = list(MODEL_REGISTRY)
        picked = st.radio(
            "Model",
            options=labels,
            index=kinds.index(default_kind),
            help="Both models are pure-Python replays of the R pipeline.",
        )
        kind = kinds[labels.index(picked)]
        f = get_model(kind)
        st.divider()
        st.markdown(
            f"""
**Model**: {f.display_name}
**Features**: {f.n_features}
**Decision threshold**: {f.youden_cut:.4f}
**Development cohort**: {_cohort(f)}
            """
        )
        st.divider()
        if st.button("Load training medians", use_container_width=True):
            for k, v in f.median_profile().items():
                st.session_state[f"in_{k}"] = v
            st.rerun()

        try:
            fx = f.load_parity_fixture()
            r = np.asarray(fx["r_prob"], dtype=float)
            Xf = np.asarray(fx["X"], dtype=float)
            hi_row = Xf[int(np.argmax(r))]
            lo_row = Xf[int(np.argmin(r))]
        except Exception:  # noqa: BLE001
            hi_row = lo_row = None
        if hi_row is not None:
            b1, b2 = st.columns(2)
            if b1.button("High-risk example", use_container_width=True):
                for k, v in zip(f.features, hi_row):
                    st.session_state[f"in_{k}"] = float(v)
                st.rerun()
            if b2.button("Low-risk example", use_container_width=True):
                for k, v in zip(f.features, lo_row):
                    st.session_state[f"in_{k}"] = float(v)
                st.rerun()
        st.divider()
        st.caption("For research use only. Not medical advice.")

    st.title("AS+AMI Mortality Risk Prediction Model")
    st.caption(
        f"{f.display_name} over {f.n_features} variables available at admission -- "
        "a pure-Python reproduction whose predictions are bit-identical to the "
        "R modelling pipeline."
    )

    tab_pred, tab_expl, tab_info = st.tabs(
        ["Prediction", "Explanation (SHAP)", "Model Info"]
    )

    with tab_pred:
        values = patient_inputs(f)
        st.divider()
        render_prediction(f, values)
    with tab_expl:
        render_explanation(f, kind)
    with tab_info:
        render_model_info(f, kind)


main()
