"""Variable display labels.

Wording follows the paper's Table 1 / `AA__mlr3_ASAMI.R:866-883` exactly.

The label tables below are a **superset** of every feature set the analysis has
used, so re-running `output/_export_model.R` against a different selected
feature set never breaks the UI. The authoritative feature list always comes
from `meta.json`, via `groups_for()` / `discrete_for()`.

Encoding provenance (verified against the same analysis script):
* `AA__mlr3_ASAMI.R:864`  sex = factor(sex, levels = c(0, 1), labels = c("Female", "Male"))
* `AA__mlr3_ASAMI.R:854-855`  BIN_VARS are all 0 = No, 1 = Yes
* `AA__mlr3_ASAMI.R:862-863`  revascularization = factor(..., 0:2, c("None", "PCI", "CABG"))
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

__all__ = [
    "FEATURE_LABELS",
    "FEATURE_NAMES",
    "CATEGORIES",
    "DISCRETE_FEATURES",
    "GROUPS",
    "groups_for",
    "discrete_for",
    "feature_label",
    "feature_plain",
    "format_value",
    "options_for",
]

# feature -> label as shown in the paper (name + unit)
FEATURE_LABELS: dict[str, str] = {
    "age": "Age (years)",
    "sex": "Sex",
    "prior_mi": "Prior MI",
    "prior_pci": "Prior PCI",
    "prior_stroke": "Prior stroke",
    "clopidogrel": "Clopidogrel",
    "statin": "Statin",
    "aspirin": "Aspirin",
    "arni": "ARNI",
    "ccb": "CCB",
    "spironolactone": "Spironolactone",
    "revascularization": "Revascularization",
    "valve_surgery": "Valve surgery",
    "hypertension": "Hypertension",
    "hyperlipidemia": "Hyperlipidemia",
    "albumin": "Albumin (g/L)",
    "alt": "ALT (U/L)",
    "ast": "AST (U/L)",
    "bnp": "BNP (pg/mL)",
    "ck_mb": "CK-MB (ng/mL)",
    "creatinine": "Creatinine (umol/L)",
    "ctn_fold": "Cardiac troponin (x URL)",
    "d_dimer": "D-dimer (mg/L FEU)",
    "glucose": "Glucose (mmol/L)",
    "hdl_c": "HDL-C (mmol/L)",
    "ldl_c": "LDL-C (mmol/L)",
    "total_cholesterol": "Total cholesterol (mmol/L)",
    "triglyceride": "Triglyceride (mmol/L)",
    "platelet": "Platelet (10^9/L)",
    "wbc": "WBC (10^9/L)",
    "uric_acid": "Uric acid (umol/L)",
    "hemoglobin": "Hemoglobin (g/L)",
    "av_mpg": "AV mean pressure gradient (mmHg)",
    "av_vmax": "AV peak velocity (m/s)",
}

# feature -> short name without the unit
FEATURE_NAMES: dict[str, str] = {
    "age": "Age",
    "sex": "Sex",
    "prior_mi": "Prior MI",
    "prior_pci": "Prior PCI",
    "prior_stroke": "Prior stroke",
    "clopidogrel": "Clopidogrel",
    "statin": "Statin",
    "aspirin": "Aspirin",
    "arni": "ARNI",
    "ccb": "CCB",
    "spironolactone": "Spironolactone",
    "revascularization": "Revascularization",
    "valve_surgery": "Valve surgery",
    "hypertension": "Hypertension",
    "hyperlipidemia": "Hyperlipidemia",
    "albumin": "Albumin",
    "alt": "ALT",
    "ast": "AST",
    "bnp": "BNP",
    "ck_mb": "CK-MB",
    "creatinine": "Creatinine",
    "ctn_fold": "Cardiac troponin",
    "d_dimer": "D-dimer",
    "glucose": "Glucose",
    "hdl_c": "HDL-C",
    "ldl_c": "LDL-C",
    "total_cholesterol": "Total cholesterol",
    "triglyceride": "Triglyceride",
    "platelet": "Platelet",
    "wbc": "WBC",
    "uric_acid": "Uric acid",
    "hemoglobin": "Hemoglobin",
    "av_mpg": "AV mean PG",
    "av_vmax": "AV Vmax",
}

# Categorical variables (0/1/2 coded) and how to word their levels.
CATEGORIES: dict[str, dict[int, str]] = {
    "sex": {0: "Female", 1: "Male"},
    "statin": {0: "No", 1: "Yes"},
    "clopidogrel": {0: "No", 1: "Yes"},
    "prior_mi": {0: "No", 1: "Yes"},
    "prior_pci": {0: "No", 1: "Yes"},
    "prior_stroke": {0: "No", 1: "Yes"},
    "aspirin": {0: "No", 1: "Yes"},
    "arni": {0: "No", 1: "Yes"},
    "ccb": {0: "No", 1: "Yes"},
    "spironolactone": {0: "No", 1: "Yes"},
    "valve_surgery": {0: "No", 1: "Yes"},
    "hypertension": {0: "No", 1: "Yes"},
    "hyperlipidemia": {0: "No", 1: "Yes"},
    "revascularization": {0: "None", 1: "PCI", 2: "CABG"},
}

# Kept for backwards compatibility; prefer discrete_for() so the set follows the
# model rather than this table.
DISCRETE_FEATURES = tuple(CATEGORIES.keys())

# Layout grouping only -- has no effect on the model. Entries that are not part
# of the loaded model are dropped by groups_for(); anything the model uses but
# this table does not list lands in a trailing "Other" group.
_GROUPS_ALL: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "Demographics & History",
        (
            "age", "sex", "prior_mi", "prior_pci", "prior_stroke",
            "hypertension", "hyperlipidemia", "revascularization",
            "valve_surgery",
        ),
    ),
    (
        "Medications",
        ("statin", "clopidogrel", "aspirin", "arni", "ccb", "spironolactone"),
    ),
    (
        "Laboratory Tests",
        (
            "uric_acid", "d_dimer", "wbc", "bnp", "albumin", "hemoglobin",
            "platelet", "creatinine", "alt", "ast", "ck_mb", "ctn_fold",
            "glucose", "hdl_c", "ldl_c", "total_cholesterol", "triglyceride",
        ),
    ),
    ("Echocardiography", ("av_vmax", "av_mpg")),
)

# Backwards-compatible alias (unfiltered layout).
GROUPS = _GROUPS_ALL


def groups_for(features: Iterable[str]) -> tuple[tuple[str, tuple[str, ...]], ...]:
    """Restrict the layout groups to `features`, appending anything unlisted.

    Guarantees every feature is rendered exactly once, in a stable order.
    """
    feats = list(features)
    seen: set[str] = set()
    out: list[tuple[str, tuple[str, ...]]] = []
    for title, names in _GROUPS_ALL:
        keep = tuple(n for n in names if n in feats)
        if keep:
            out.append((title, keep))
            seen.update(keep)
    rest = tuple(n for n in feats if n not in seen)
    if rest:
        out.append(("Other", rest))
    return tuple(out)


def discrete_for(
    features: Sequence[str], discrete_values: dict | None = None
) -> tuple[str, ...]:
    """Which features to render as dropdowns.

    Prefers the model metadata (`discrete_values` from `meta.json`, which the R
    export fills with the observed levels of any column having <= 6 of them),
    falling back to the wording table when metadata is unavailable.
    """
    if discrete_values:
        return tuple(n for n in features if discrete_values.get(n))
    return tuple(n for n in features if n in CATEGORIES)


def feature_label(name: str) -> str:
    """Label with unit, for inputs and axes."""
    return FEATURE_LABELS.get(name, name)


def feature_plain(name: str) -> str:
    """Short label without the unit."""
    return FEATURE_NAMES.get(name, name)


def format_value(name: str, value) -> str:
    """Render a raw value the way a clinician would read it."""
    if name in CATEGORIES:
        try:
            return CATEGORIES[name].get(int(round(float(value))), str(value))
        except (TypeError, ValueError):
            return str(value)
    try:
        v = float(value)
    except (TypeError, ValueError):
        return str(value)
    if abs(v) >= 1000:
        return f"{v:,.0f}"
    if abs(v) >= 10:
        return f"{v:.1f}"
    if abs(v) >= 1:
        return f"{v:.2f}"
    return f"{v:.4g}"


def options_for(name: str, levels: Sequence | None = None) -> list[tuple[str, int]]:
    """Dropdown options as [(display text, value), ...].

    `levels` (the model's observed levels) wins when supplied so that a feature
    absent from CATEGORIES still yields sensible 0/1 wording.
    """
    if levels:
        vals = sorted(int(round(float(v))) for v in levels)
    elif name in CATEGORIES:
        vals = sorted(CATEGORIES[name])
    else:
        vals = [0, 1]
    wording = CATEGORIES.get(name, {})
    return [(wording.get(v, str(v)), v) for v in vals]
