"""
evaluate.py

Metrics of the analysis plan (protocol/analysis_plan.md, section 4): sensitivity at fixed specificity,
patient-level bootstrap interval, calibration intercept/slope and ECE, subgroup report.
All of it works on pooled out-of-fold predictions, never on a single fold.
Cohorts without patient identifiers (Chapman-Shaoxing headers carry none): pass record ids
as `patient_ids`; the interval is then record-level and must be reported as such.
"""
import numpy as np
from sklearn.metrics import roc_curve
from sklearn.linear_model import LogisticRegression


def sensitivity_at_specificity(y_true, y_prob, target_specificity=0.95):
    y_true = np.asarray(y_true); y_prob = np.asarray(y_prob)
    if y_true.sum() == 0 or y_true.sum() == len(y_true):
        return np.nan, np.nan
    fpr, tpr, thr = roc_curve(y_true, y_prob)
    spec = 1 - fpr
    ok = spec >= target_specificity
    idx = np.where(ok)[0][np.argmax(tpr[ok])] if ok.any() else np.argmax(spec)
    return tpr[idx], thr[idx]


def bootstrap_sensitivity_ci(y_true, y_prob, patient_ids, target_specificity=0.95, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true); y_prob = np.asarray(y_prob); pid = np.asarray(patient_ids)
    uniq, inv = np.unique(pid, return_inverse=True)
    members = [np.where(inv == k)[0] for k in range(len(uniq))]
    vals = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(uniq), len(uniq))
        idx = np.concatenate([members[k] for k in pick])         # resample whole patients, with replacement
        if len(idx) < 10:
            continue
        s, _ = sensitivity_at_specificity(y_true[idx], y_prob[idx], target_specificity)
        if not np.isnan(s):
            vals.append(s)
    point, _ = sensitivity_at_specificity(y_true, y_prob, target_specificity)
    if not vals:
        return np.nan, (np.nan, np.nan)
    return point, tuple(np.percentile(vals, [2.5, 97.5]))


def calibration_metrics(y_true, y_prob, n_bins=10):
    y_true = np.asarray(y_true); y_prob = np.asarray(y_prob)
    p = np.clip(y_prob, 1e-6, 1 - 1e-6)
    logit = np.log(p / (1 - p)).reshape(-1, 1)
    slope = intercept = np.nan
    if len(np.unique(y_true)) == 2:
        lr = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000).fit(logit, y_true)
        slope, intercept = lr.coef_[0][0], lr.intercept_[0]
    edges = np.linspace(0, 1, n_bins + 1); ece = 0.0
    for i in range(n_bins):
        m = (y_prob >= edges[i]) & ((y_prob < edges[i + 1]) if i < n_bins - 1 else (y_prob <= edges[i + 1]))
        if m.any():
            ece += m.mean() * abs(y_prob[m].mean() - y_true[m].mean())
    return {"calibration_slope": slope, "calibration_intercept": intercept, "ece": ece}


def subgroup_report(y_true, y_prob, patient_ids, age, age_threshold=65, target_specificity=0.95, n_boot=2000, seed=0):
    age = np.asarray(age); out = {}
    for label, mask in (("overall", np.ones(len(age), bool)), (f"age>={age_threshold}", age >= age_threshold)):
        yt, yp, pid = np.asarray(y_true)[mask], np.asarray(y_prob)[mask], np.asarray(patient_ids)[mask]
        s, (lo, hi) = bootstrap_sensitivity_ci(yt, yp, pid, target_specificity, n_boot, seed)
        out[label] = {"n": int(mask.sum()), "n_positive": int(yt.sum()), "sensitivity": s, "ci_lo": lo, "ci_hi": hi,
                      **calibration_metrics(yt, yp)}
    return out


if __name__ == "__main__":
    rng = np.random.default_rng(0); n = 3000
    age = rng.integers(18, 95, n); y = (rng.random(n) < 0.15 + 0.005 * age).astype(int)
    p = np.clip(y * 0.7 + 0.15 + rng.normal(0, 0.15 + 0.002 * age, n), 0.01, 0.99)
    rep = subgroup_report(y, p, np.arange(n), age, n_boot=200)
    for k, v in rep.items():
        print(k, {a: (round(b, 3) if isinstance(b, float) else b) for a, b in v.items()})
    assert rep["age>=65"]["sensitivity"] < rep["overall"]["sensitivity"]
    print("evaluate.py OK")
