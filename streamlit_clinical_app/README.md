# AS+AMI Mortality Risk Prediction Model

A Streamlit web app that predicts in-hospital mortality for patients with aortic
stenosis (AS) complicated by acute myocardial infarction (AMI), using 14
variables available at admission.

The **delivered model is a logistic regression** (R `classif.log_reg`, E53),
selected over the 680-tree `ranger` probability forest because it is better
calibrated (slope 1.114 vs 1.387), has the lowest Brier score (0.1273 vs 0.1286),
a monotone seven-band risk gradient (2.3% → 55.2%), and the best net benefit over
the clinically relevant 20–60% threshold range — at no detectable cost in
discrimination (ΔAUC = −0.0043, DeLong p = 0.836). The forest ships alongside it
as a comparison model and can be switched from the sidebar.

Both are produced by the R modelling pipeline. **The Python app does not retrain
anything** — it replays the exported coefficients and trees, so its predictions
are *bit-identical* to the R pipeline.

---

## Quick start (local)

```bash
cd streamlit_clinical_app
pip install -r requirements.txt
streamlit run app.py
```

Then open <http://localhost:8501>.

Verified locally with Python 3.13.5, Streamlit 1.45+, numpy 2.x, pandas 2.x,
plotly 5.x. No R installation is required to run the app.

---

## Deploying to Streamlit Community Cloud

1. Push **the contents of this directory as the repository root** (so that
   `app.py` sits at the top level, not inside a subfolder).
2. Go to <https://share.streamlit.io> → *New app* → pick the repository.
3. Set **Main file path** to `app.py`.
4. Under *Advanced settings* choose **Python 3.12**.
5. Deploy. First build takes ~2 minutes.

Notes:

* **`packages.txt` is not needed.** No apt packages, no `r-base`, no R packages.
  The only dependencies are the four Python packages in `requirements.txt`.
* `model/*.json` totals roughly 2.3 MB and is committed to the repository, so the
  app is fully self-contained.
* This directory never reads from outside itself -- it does not depend on
  `output/` or `mlr3_output/`. Mentions of those paths appear only in comments
  and documentation, recording where the artefacts came from.

> **Why not run the original R Shiny app in the cloud instead?**
> Streamlit Community Cloud exposes exactly one port per container. An R
> `shiny::runApp()` bound to container-local `localhost:3838` is unreachable from
> a remote browser, and installing the `mlr3verse` stack (hundreds of R packages)
> from source on Debian would not finish inside the build timeout. Replaying the
> exported forest in pure Python removes R from the runtime entirely.

---

## Project structure

```
streamlit_clinical_app/
├── app.py                    # Streamlit entry point (3 tabs)
├── requirements.txt
├── .streamlit/config.toml    # theme
├── model/
│   ├── logistic.json         # intercept + 14 coefficients + per-SD odds ratios
│   ├── forest.json           # 680 trees (1.14 MB)
│   ├── background.json       # 200-row SHAP background distribution
│   ├── parity_logistic.json  # 25 reference rows + R's own glm probabilities
│   ├── parity_forest.json    # 25 reference rows + ranger's own probabilities
│   └── meta.json             # feature list/order, ranges, per-model thresholds & metrics
├── src/
│   ├── logistic.py           # LogisticModel: loads logistic.json, evaluates the linear predictor
│   ├── forest.py             # RangerForest: loads forest.json, replays the trees
│   ├── explain.py            # sampled permutation SHAP (model-agnostic)
│   └── labels.py             # variable labels + categorical encodings
└── tests/
    ├── test_app_smoke.py     # AppTest smoke + parity assertion
    └── test_shap.py          # SHAP local accuracy + direction sanity
```

### Tabs

| Tab | Contents |
| --- | --- |
| **Prediction** | 14 inputs (9 continuous, 5 categorical), gauge, risk stratum, threshold explanation, holdout performance table |
| **Explanation (SHAP)** | Per-patient SHAP bar chart, baseline vs patient risk, ranked contribution table |
| **Model Info** | Provenance, coefficient table (logistic) or hyperparameters (forest), feature table, metrics, research-use disclaimer |

The sidebar carries a **model selector** (logistic regression by default), the
decision threshold, the development-cohort size, and buttons that load the
training medians or the highest/lowest-risk fixture patients.

---

## How the export works

`output/_export_model.R` (kept with the modelling pipeline, not in this folder)
writes both models.

### Logistic regression (`logistic.json`)

Because the model is linear there is nothing to replay: the export writes the
intercept and the 14 coefficients, plus per-SD odds ratios for interpretation.
R verifies `1 / (1 + exp(-(intercept + X β)))` against mlr3's own prediction
over all 1,052 training rows before writing.

### Random forest (`forest.json`)

Every tree is dumped with `ranger::treeInfo()`:

* leaves are rewritten with `left = right = -1`, removing the ambiguity in
  ranger's raw `child.nodeIDs` where pointer `0` means both "no child" and "node
  id 0" (the root);
* `feature` is a 0-based column index and is `-1` at leaves;
* the split rule is `x[feature] <= threshold` → go left;
* every leaf carries its own `pred.Death` probability.

Every `toJSON()` call in the export uses `digits = I(17)`, i.e. `sprintf("%.17g")`.
This is **not** cosmetic -- the round number `digits = NA` is silently lossy:

```
> jsonlite::toJSON(12.219999999999998863, digits = NA)   # a split threshold
["12.22"]
> jsonlite::toJSON(12.220000000000000639, digits = NA)   # a patient's value
["12.22"]
```

Those two doubles are one unit in the last place apart, and the patient sits that
one ULP *above* the threshold, so ranger sends them down the right-hand branch.
With `digits = NA` the export collapses them to the same decimal, Python evaluates
`x <= threshold` as true, takes the left branch, and casts the opposite vote.
`digits = I(17)` round-trips all 26,170 split values and 26,850 leaf
probabilities exactly.

**Self-check built into the export.** The R script replays all 1,052 training rows
through all 680 exported trees and requires the maximum absolute deviation from
ranger's own per-tree predictions to be `< 1e-12`. Measured value: **exactly 0**.

**Independent Python-side check.** `tests/test_app_smoke.py` compares both
replays against the fixtures R shipped (`parity_logistic.json`,
`parity_forest.json` -- 25 rows carrying R's own probabilities):

| Check | Logistic | Random forest |
| --- | --- | --- |
| Max abs deviation over 25 rows | **1.665e-16** | **5.551e-17** |
| Classification mismatch at threshold 0.5 | **0 / 25** | **0 / 25** |
| Latency | ~µs / patient (closed form) | 0.49 ms / patient (22 ms for 200) |

Both residuals are a couple of double-epsilons and come purely from summation
order (R accumulates tree by tree; numpy vectorises), not from a modelling
difference.

---

## SHAP

`src/explain.py` implements **sampled permutation SHAP** directly, rather than
using the `shap` package. `shap.TreeExplainer` only accepts scikit-learn tree
objects, so it could not explain the logistic regression at all, and re-wrapping
the ranger trees risks silently diverging from the model under test.

The implementation is model-agnostic: it only needs `predict_proba_death`,
`load_background` and `features`, so the sidebar model selector changes what is
explained without touching the algorithm.

This is the same algorithm as R's `iml::permshap` (the `Sampling version of
permutation SHAP` path taken in the modelling run), so explanations here share
their method with the paper's SHAP figures -- which are computed on the
**logistic regression**, the same model the sidebar defaults to (R setting
`RUN_SHAP`, change E54). Switching the sidebar to *Random forest* explains the
comparison model instead, so the in-app numbers will then differ from the paper.

* Defaults: `n_perm = 64`, `bg_size = 50` → about **0.03 s** per patient for the
  logistic model and **16 s** for the forest
* Local accuracy: `|base + Σφ − f(x)| = 2.8e-17` (logistic), `1.4e-17` (forest)
* Top-5 feature ranking is identical across seeds 2026 / 7 / 99

---

## Privacy note

`model/background.json` contains the 14 feature values of **200 real patients**
drawn from the development cohort. There are no identifiers, names, or dates, but
if the cohort data cannot be made public, replace this file with synthetic values
or with per-feature quantiles before pushing the repository. The app only needs a
reasonable background distribution for SHAP; it does not need real rows.

---

## Disclaimer

**For research use only.** The model was developed on a single-centre
retrospective cohort (1,052 patients, 194 deaths) and has not been externally
validated. It does not constitute medical advice and must not be used for
clinical decision-making.
