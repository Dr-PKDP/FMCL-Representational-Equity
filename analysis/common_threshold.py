"""
common_threshold.py: age-disparity quantities at ONE operating threshold per model (deployment-oriented complement to the
subgroup-specific estimand of ecg_analysis.py).

    python3 common_threshold.py [--task ecg|derm] --results <results/ecg> --cache-dir <cache> --out-dir <analysis out> \
        --suites primary val_common ... [--boot 2000] [--folds ...]

ecg_analysis.py sets the 95%-specificity threshold within each age subgroup, so the age gap it reports is the difference of
subgroup sensitivities at subgroup-specific thresholds. Here every (arm, seed) model has one threshold, the score at which the
specificity over ALL test patients is 95%, and the sensitivity and specificity are then read within the age groups at that threshold:
    overall     sensitivity at the common threshold (equal to the overall estimate of ecg_analysis.py)
    age65       sensitivity at the common threshold among patients aged 65 and over
    under65     sensitivity at the common threshold among patients under 65
    gap         age65 minus under65
    spec65, specunder65   specificity within the two groups at the common threshold
The patient resampling is the one of ecg_analysis.py (same generator seed), the threshold is re-estimated on every resample, and
arms are paired through the shared resample. Contrasts and arms use the same pairs as ecg_analysis.py.
Outputs: <out-dir>/commonthr/<suite>_arms.csv and <suite>_contrasts.csv
"""
import argparse, os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ecg_analysis as EA

QUANT = ["overall", "age65", "under65", "gap", "spec65", "specunder65"]


def common_threshold(score, y, W, masks, spec=0.95, chunk=250):
    """Sensitivity and specificity in each mask at the threshold giving specificity `spec` over all records, for each weight row of W."""
    order = np.argsort(-score, kind="stable"); yo = y[order].astype(np.float32)
    mo = {k: m[order].astype(np.float32) for k, m in masks.items()}
    B = W.shape[0]; sens = {k: np.empty(B) for k in masks}; spc = {k: np.empty(B) for k in masks}
    for a in range(0, B, chunk):
        w = W[a:a + chunk][:, order]; r = np.arange(w.shape[0])
        cn = np.cumsum(w * (1 - yo), axis=1); k = (cn <= (1 - spec) * cn[:, -1:]).sum(axis=1) - 1; kk = np.clip(k, 0, None)
        for name, m in mo.items():
            cp = np.cumsum(w * yo * m, axis=1); cnm = np.cumsum(w * (1 - yo) * m, axis=1)
            sens[name][a:a + chunk] = np.where(k >= 0, cp[r, kk] / np.maximum(cp[:, -1], 1e-12), 0.0)
            spc[name][a:a + chunk] = 1.0 - np.where(k >= 0, cnm[r, kk] / np.maximum(cnm[:, -1], 1e-12), 0.0)
    return sens, spc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=list(EA.TASKS), default="ecg")
    ap.add_argument("--results", required=True); ap.add_argument("--cache-dir", required=True); ap.add_argument("--out-dir", required=True)
    ap.add_argument("--suites", nargs="+", required=True); ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--folds", type=int, nargs="+", default=None)
    a = ap.parse_args()
    EA.TASK.clear(); EA.TASK.update(EA.TASKS[a.task])
    meta = pd.read_csv(os.path.join(a.cache_dir, EA.TASK["meta"])); meta = meta[meta.loaded == 1].sort_values(EA.TASK["rid"]).reset_index(drop=True)
    if a.folds is not None: meta = meta[meta.strat_fold.isin(a.folds)].reset_index(drop=True)
    canon = meta[EA.TASK["rid"]].values; y = meta[EA.TASK["label"]].values.astype(int); age = meta.age_capped.values
    masks = {"overall": np.ones(len(meta), bool), "age65": age >= 65, "under65": age < 65}
    pat = pd.factorize(meta.patient_id.values)[0]; P = pat.max() + 1
    C = np.random.default_rng(20261107).multinomial(P, np.ones(P) / P, size=a.boot).astype(np.float32)
    out = os.path.join(a.out_dir, "commonthr"); os.makedirs(out, exist_ok=True)
    for suite in a.suites:
        preds, _, kind = EA.merged_preds(a.results, suite, canon, a.folds)
        labels = sorted(preds); pairs = EA.pairs_for(kind, labels)
        if "repr_aware" in preds and "repr_aware@0.75" in preds: pairs = pairs + [("repr_aware", "repr_aware@0.75")]
        res = {}
        for lab in labels:
            for sk, p in preds[lab].items():
                if np.isnan(p).any(): raise SystemExit(f"{suite} {lab} {sk}: records not predicted")
                s0, c0 = common_threshold(p, y, np.ones((1, len(y)), np.float32), masks)
                sb, cb = common_threshold(p, y, C[:, pat], masks)
                d = {}
                for q in ("overall", "age65", "under65"): d[q] = (s0[q][0], sb[q])
                d["gap"] = (s0["age65"][0] - s0["under65"][0], sb["age65"] - sb["under65"])
                d["spec65"] = (c0["age65"][0], cb["age65"]); d["specunder65"] = (c0["under65"][0], cb["under65"])
                for q in QUANT: res[(lab, sk, q)] = d[q]
        arows = []
        for lab in labels:
            sks = sorted(preds[lab]); r = dict(arm=lab, n_seeds=len(sks))
            for q in QUANT:
                pts = np.array([res[(lab, s, q)][0] for s in sks]); bs = np.mean([res[(lab, s, q)][1] for s in sks], axis=0)
                lo, hi = np.percentile(bs, [2.5, 97.5])
                r.update({q: pts.mean(), f"{q}_lo": lo, f"{q}_hi": hi, f"{q}_seedsd": pts.std(ddof=1) if len(pts) > 1 else np.nan})
            arows.append(r)
        crows = []
        for x, z in pairs:
            if x not in preds or z not in preds: continue
            common = sorted(set(preds[x]) & set(preds[z]))
            for q in QUANT:
                ds = np.array([res[(x, s, q)][0] - res[(z, s, q)][0] for s in common])
                bs = np.mean([res[(x, s, q)][1] - res[(z, s, q)][1] for s in common], axis=0)
                lo, hi = np.percentile(bs, [2.5, 97.5])
                crows.append(dict(a=x, b=z, quantity=q, n_seeds=len(common), diff=ds.mean(), lo95=lo, hi95=hi,
                                  seedsd=ds.std(ddof=1) if len(common) > 1 else np.nan,
                                  p_boot=min(1.0, 2 * min((bs <= 0).mean(), (bs >= 0).mean()) + 1.0 / (a.boot + 1))))
        pd.DataFrame(arows).to_csv(os.path.join(out, f"{suite}_arms.csv"), index=False)
        con = pd.DataFrame(crows); con.to_csv(os.path.join(out, f"{suite}_contrasts.csv"), index=False)
        pd.set_option("display.width", 220, "display.max_columns", 30)
        print(f"[{suite}] common threshold"); print(pd.DataFrame(arows)[["arm", "n_seeds", "overall", "age65", "under65", "gap", "spec65", "specunder65"]].round(3).to_string(index=False))
        print(con[con.quantity.isin(["age65", "under65", "gap"])][["a", "b", "quantity", "diff", "lo95", "hi95"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
