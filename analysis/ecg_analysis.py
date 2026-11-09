"""
ecg_analysis.py: Task A analysis of the pooled out-of-fold predictions (protocol/analysis_plan.md).

    python3 ecg_analysis.py --results <results/ecg> --cache-dir <cache> --out-dir <analysis out> \
        --suites primary [frontier] [sens_none ...] [--boot 2000]

For every suite it
  * pools the ten test folds of each (arm, seed) into one set of out-of-fold predictions (checks every record is
    predicted exactly once),
  * computes sensitivity at 95% specificity overall, 65+, 75+ (threshold set within the subgroup), AUC, and 65+ calibration,
  * draws ONE patient-level bootstrap (shared by all arms and seeds, so contrasts are paired) and reports each arm's mean
    over seeds with its 95% interval, and the contrasts of the plan with 95% and 90% intervals, a bootstrap p value, an
    exact sign-flip p value across seeds, the equivalence verdict (TOST, margin 4 points, primary contrast only) and
    Benjamini-Hochberg adjusted p values over the secondary family,
  * summarises the job diagnostics (pool size, recruits, composition, steps, participations).
Outputs: <out-dir>/<suite>_arms.csv, _contrasts.csv, _diagnostics.csv, _summary.json
"""
import argparse, glob, itertools, json, os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "modelling"))
from evaluate import calibration_metrics

SUBGROUPS = ["overall", "age65", "age75"]
MARGIN = 0.04
PRIMARY = ("availability", "uniform", "age65")                      # H1
SECONDARY = [("size_matched", "uniform"), ("availability", "size_matched"), ("repr_aware", "availability"),
             ("post_hoc", "availability"), ("repr_aware", "uniform")]


def label_of(job):
    if job["arm"] == "repr_aware" and abs(job.get("boost", 1.0) - 1.0) > 1e-9:
        return f"repr_aware@{job['boost']:g}"
    return job["arm"]


TASKS = {"ecg": dict(meta="ptbxl_meta.csv", rid="ecg_id", label="has_cd"),
         "derm": dict(meta="pad_meta.csv", rid="row_idx", label="has_malignant")}
TASK = dict(TASKS["ecg"])


def load_suite(results, suite, canon_ecg):
    """-> dict label -> dict seed -> prob array aligned to canon_ecg; plus diagnostics DataFrame."""
    idx = {int(e): i for i, e in enumerate(canon_ecg)}
    jobs = {}
    diags = []
    for jp in glob.glob(os.path.join(results, suite, "*.json")):
        d = json.load(open(jp)); npz = jp[:-5] + ".npz"
        if not os.path.exists(npz): continue
        z = np.load(npz); j = d["job"]
        key = (label_of(j), j["init_seed"], j["elig_seed"])
        arr = jobs.setdefault(key, np.full(len(canon_ecg), np.nan))
        pos = np.array([idx[int(e)] for e in z[TASK["rid"]]])
        if not np.isnan(arr[pos]).all():
            raise SystemExit(f"duplicate predictions for {key} fold {j['fold']}")
        arr[pos] = z["y_prob"]
        diags.append(dict(label=key[0], seed=key[1], fold=j["fold"], n_eligible=d["n_eligible"], n_recruited=d["n_recruited"],
                          share65=d["share65"], stream65=d["stream65"], steps=d["steps"], participations=d["participations"],
                          samples=d["samples_processed"], seconds=d["seconds"], best_val_loss=d["best_val_loss"]))
    out = {}
    for (lab, s1, s2), arr in jobs.items():
        out.setdefault(lab, {})[(s1, s2)] = arr
    return out, pd.DataFrame(diags)


def sens_weighted(score, y, W, spec=0.95, chunk=250):
    """Sensitivity at specificity >= spec for each bootstrap weight row of W (B, n); W=ones for the point estimate."""
    order = np.argsort(-score, kind="stable"); yo = y[order].astype(np.float32)
    out = np.empty(W.shape[0])
    for a in range(0, W.shape[0], chunk):
        w = W[a:a + chunk][:, order]
        cp = np.cumsum(w * yo, axis=1); cn = np.cumsum(w * (1 - yo), axis=1)
        ok = cn <= (1 - spec) * cn[:, -1:]
        k = ok.sum(axis=1) - 1
        tpr = np.where(k >= 0, cp[np.arange(len(k)), np.clip(k, 0, None)] / np.maximum(cp[:, -1], 1e-12), 0.0)
        out[a:a + chunk] = tpr
    return out


def auc(y, p):
    from scipy.stats import rankdata
    r = rankdata(p); n1 = y.sum(); n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def bh(p):
    p = np.asarray(p, float); n = len(p); o = np.argsort(p); q = np.empty(n)
    prev = 1.0
    for rank, i in zip(range(n, 0, -1), o[::-1]):
        prev = min(prev, p[i] * n / rank); q[i] = prev
    return q


def signflip_p(d, n_mc=200000, seed=0):
    d = np.asarray(d, float); S = len(d); obs = abs(d.mean())
    if S <= 20:
        signs = ((np.arange(2 ** S)[:, None] >> np.arange(S)) & 1) * 2 - 1
        stat = np.abs((signs * d).mean(axis=1))
    else:
        signs = np.random.default_rng(seed).choice([-1, 1], size=(n_mc, S)); stat = np.abs((signs * d).mean(axis=1))
    return float((stat >= obs - 1e-12).mean())


def analyse(results, cache_dir, suite, boot, out_dir, seed=20261107):
    meta = pd.read_csv(os.path.join(cache_dir, TASK["meta"])); meta = meta[meta.loaded == 1].sort_values(TASK["rid"]).reset_index(drop=True)
    canon = meta[TASK["rid"]].values; y = meta[TASK["label"]].values.astype(int)
    age = meta.age_capped.values; sub = {"overall": np.ones(len(meta), bool), "age65": age >= 65, "age75": age >= 75}
    pat = pd.factorize(meta.patient_id.values)[0]; P = pat.max() + 1
    preds, diag = load_suite(results, suite, canon)
    if not preds: raise SystemExit(f"no results for suite {suite}")
    pre = "derm_" if suite.startswith("derm_") else ""; kind = suite[len(pre):]       # suite kind without the task prefix
    if kind in ("volume_control", "volume", "frontier"):
        base, bdiag = load_suite(results, pre + "primary", canon)          # same seeds, folds and resampling draws as the primary suite
        preds.update({k: v for k, v in base.items() if k in ("availability", "repr_aware")})
        diag = pd.concat([diag, bdiag[bdiag.label.isin(["availability", "repr_aware"])]], ignore_index=True)
    labels = sorted(preds)
    for lab in labels:
        for k, arr in preds[lab].items():
            if np.isnan(arr).any(): raise SystemExit(f"{suite} {lab} seed {k}: {int(np.isnan(arr).sum())} records not predicted")
    print(f"[{suite}] arms: {labels}; seeds per arm: {[len(preds[l]) for l in labels]}", flush=True)

    rng = np.random.default_rng(seed)
    C = rng.multinomial(P, np.ones(P) / P, size=boot).astype(np.float32)           # patient resampling counts (boot, P)
    sens = {}                                                                           # (label, seedkey, sub) -> (point, boot array)
    cal, aucs = {}, {}
    for lab in labels:
        for sk, p in preds[lab].items():
            for g in SUBGROUPS:
                m = sub[g]
                pt = sens_weighted(p[m], y[m], np.ones((1, m.sum()), np.float32))[0]
                bs = sens_weighted(p[m], y[m], C[:, pat[m]])
                sens[(lab, sk, g)] = (pt, bs)
            m = sub["age65"]
            cal[(lab, sk)] = calibration_metrics(y[m], p[m])
            aucs[(lab, sk)] = (auc(y, p), auc(y[sub["age65"]], p[sub["age65"]]))

    rows = []
    for lab in labels:
        sks = sorted(preds[lab])
        r = dict(arm=lab, n_seeds=len(sks))
        for g in SUBGROUPS:
            pts = np.array([sens[(lab, s, g)][0] for s in sks]); bs = np.mean([sens[(lab, s, g)][1] for s in sks], axis=0)
            lo, hi = np.percentile(bs, [2.5, 97.5])
            r.update({f"sens95_{g}": pts.mean(), f"sens95_{g}_lo": lo, f"sens95_{g}_hi": hi, f"sens95_{g}_seedsd": pts.std(ddof=1) if len(pts) > 1 else np.nan})
        r["auc_overall"] = np.mean([aucs[(lab, s)][0] for s in sks]); r["auc_age65"] = np.mean([aucs[(lab, s)][1] for s in sks])
        for c in ("calibration_slope", "calibration_intercept", "ece"):
            r[f"{c}_age65"] = np.mean([cal[(lab, s)][c] for s in sks])
        rows.append(r)
    arms = pd.DataFrame(rows)

    crows = []
    pairs = ([(PRIMARY[0], PRIMARY[1])] + SECONDARY) if kind == "primary" or "availability" in labels else []
    if kind == "frontier":
        pairs = [(l, "availability") for l in labels if l != "availability"]
    if kind in ("volume_control", "volume"):
        pairs = [("recruit_random", "availability"), ("repr_aware", "recruit_random"), ("repr_aware", "availability")]
    for a, b in pairs:
        if a not in preds or b not in preds: continue
        common = sorted(set(preds[a]) & set(preds[b]))
        for g in SUBGROUPS:
            d_seed = np.array([sens[(a, s, g)][0] - sens[(b, s, g)][0] for s in common])
            bs = np.mean([sens[(a, s, g)][1] - sens[(b, s, g)][1] for s in common], axis=0)
            pboot = min(1.0, 2 * min((bs <= 0).mean(), (bs >= 0).mean()) + 1.0 / (boot + 1))
            lo95, hi95 = np.percentile(bs, [2.5, 97.5]); lo90, hi90 = np.percentile(bs, [5, 95])
            crows.append(dict(a=a, b=b, subgroup=g, n_seeds=len(common), diff=d_seed.mean(), lo95=lo95, hi95=hi95, lo90=lo90, hi90=hi90,
                              seedsd=d_seed.std(ddof=1) if len(common) > 1 else np.nan, p_boot=pboot, p_signflip=signflip_p(d_seed) if len(common) > 1 else np.nan,
                              equivalent_pm4=bool(lo90 > -MARGIN and hi90 < MARGIN), primary=((a, b, g) == PRIMARY)))
    con = pd.DataFrame(crows)
    if len(con):
        fam = (~con.primary) & (con.subgroup != "overall")
        con["p_boot_bh"] = np.nan
        if fam.any(): con.loc[fam, "p_boot_bh"] = bh(con.loc[fam, "p_boot"].values)

    # Age-disparity quantities (not part of the multiplicity family): sensitivity below 65 and the age gap (65+ minus under 65),
    # per arm and as contrasts between arms, plus the direct contrast of the two highest boost levels when both are present.
    under = age < 65
    for lab in labels:
        for sk, p in preds[lab].items():
            pt_u = sens_weighted(p[under], y[under], np.ones((1, under.sum()), np.float32))[0]
            bs_u = sens_weighted(p[under], y[under], C[:, pat[under]])
            sens[(lab, sk, "under65")] = (pt_u, bs_u)
            sens[(lab, sk, "gap")] = (sens[(lab, sk, "age65")][0] - pt_u, sens[(lab, sk, "age65")][1] - bs_u)
    EXTRA = ["overall", "age65", "age75", "under65", "gap"]
    erows = []
    for lab in labels:
        sks = sorted(preds[lab]); r = dict(arm=lab, n_seeds=len(sks))
        for g in ("under65", "gap"):
            pts = np.array([sens[(lab, s, g)][0] for s in sks]); bs = np.mean([sens[(lab, s, g)][1] for s in sks], axis=0)
            lo, hi = np.percentile(bs, [2.5, 97.5])
            r.update({f"{g}": pts.mean(), f"{g}_lo": lo, f"{g}_hi": hi, f"{g}_seedsd": pts.std(ddof=1) if len(pts) > 1 else np.nan})
        erows.append(r)
    epairs = list(pairs)
    if "repr_aware" in preds and "repr_aware@0.75" in preds: epairs.append(("repr_aware", "repr_aware@0.75"))
    ecrows = []
    for a, b in epairs:
        if a not in preds or b not in preds: continue
        common = sorted(set(preds[a]) & set(preds[b]))
        for g in EXTRA:
            d_seed = np.array([sens[(a, s, g)][0] - sens[(b, s, g)][0] for s in common])
            bs = np.mean([sens[(a, s, g)][1] - sens[(b, s, g)][1] for s in common], axis=0)
            lo95, hi95 = np.percentile(bs, [2.5, 97.5]); lo90, hi90 = np.percentile(bs, [5, 95])
            ecrows.append(dict(a=a, b=b, quantity=g, n_seeds=len(common), diff=d_seed.mean(), lo95=lo95, hi95=hi95, lo90=lo90, hi90=hi90,
                               seedsd=d_seed.std(ddof=1) if len(common) > 1 else np.nan,
                               p_boot=min(1.0, 2 * min((bs <= 0).mean(), (bs >= 0).mean()) + 1.0 / (boot + 1)),
                               p_signflip=signflip_p(d_seed) if len(common) > 1 else np.nan))

    dg = diag.groupby("label").agg(n_eligible=("n_eligible", "mean"), n_recruited=("n_recruited", "mean"), pool_share65=("share65", "mean"),
                                   stream_share65=("stream65", "mean"), steps=("steps", "mean"), participations=("participations", "mean"),
                                   seconds=("seconds", "mean"), jobs=("seed", "size")).reset_index()
    os.makedirs(out_dir, exist_ok=True)
    arms.to_csv(os.path.join(out_dir, f"{suite}_arms.csv"), index=False); con.to_csv(os.path.join(out_dir, f"{suite}_contrasts.csv"), index=False)
    dg.to_csv(os.path.join(out_dir, f"{suite}_diagnostics.csv"), index=False)
    os.makedirs(os.path.join(out_dir, "agegap"), exist_ok=True)
    pd.DataFrame(erows).to_csv(os.path.join(out_dir, "agegap", f"{suite}_arms.csv"), index=False)
    pd.DataFrame(ecrows).to_csv(os.path.join(out_dir, "agegap", f"{suite}_contrasts.csv"), index=False)
    summ = dict(suite=suite, boot=boot, n_records=int(len(meta)), n_patients=int(P), positives=int(y.sum()), positives_65=int(y[sub["age65"]].sum()),
                positives_75=int(y[sub["age75"]].sum()), arms=labels, bootstrap_seed=seed)
    json.dump(summ, open(os.path.join(out_dir, f"{suite}_summary.json"), "w"), indent=2)
    pd.set_option("display.width", 220, "display.max_columns", 30)
    print(arms[["arm", "n_seeds", "sens95_overall", "sens95_age65", "sens95_age65_lo", "sens95_age65_hi", "sens95_age75", "auc_overall"]].round(3).to_string(index=False))
    if len(con): print(con[con.subgroup != "overall"][["a", "b", "subgroup", "diff", "lo95", "hi95", "lo90", "hi90", "p_boot", "p_boot_bh", "equivalent_pm4"]].round(3).to_string(index=False))
    print(dg.round(1).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True); ap.add_argument("--cache-dir", required=True); ap.add_argument("--out-dir", required=True)
    ap.add_argument("--task", choices=list(TASKS), default="ecg"); ap.add_argument("--suites", nargs="+", default=["primary"]); ap.add_argument("--boot", type=int, default=2000)
    a = ap.parse_args()
    TASK.update(TASKS[a.task])
    for s in a.suites: analyse(a.results, a.cache_dir, s, a.boot, a.out_dir)
