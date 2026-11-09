"""
external_analysis.py: external cohorts (CODE-15%, Chapman plus Ningbo) scored by the PTB-XL-trained primary models.

    python3 external_analysis.py --scores <analysis dir with ext_*.npy> --cache-dir <cache> --out-dir <analysis dir> [--boot 1000]

For each (arm, seed) the ten fold models are averaged (probability mean), giving one score per record; metrics are
sensitivity at 95% specificity (threshold set within the subgroup), AUC overall, for the subgroups
overall / age>=65 / age>=75 (and, for CODE-15%, age>=65 restricted to full-length 4,096-sample traces; for Chapman plus Ningbo,
age>=65 within each inferred hospital). The bootstrap resamples PATIENTS (CODE-15%) or RECORDS (Chapman plus Ningbo has no
patient identifier), with one resample shared by all arms and seeds. Contrasts as in the plan. Records with loaded == 0
(and, for the age strata of Chapman plus Ningbo, age_valid == 0) are excluded.
Equivalence verdicts (margin 4 points, 90% interval) are produced for CODE-15% only; Chapman plus Ningbo is descriptive.
"""
import argparse, json, os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ecg_analysis import sens_weighted, auc, bh, signflip_p, MARGIN, SECONDARY

PAIRS = [("availability", "uniform")] + SECONDARY


def run_cohort(name, scores, models, meta, boot, out_dir, seed=20261107):
    ok = (meta.loaded == 1).values
    if name == "chapman":
        ok_age = ok & (meta.age_valid == 1).values
        pid = meta.record.values
    else:
        ok_age = ok; pid = meta.patient_id.values
    y = meta.has_cd.values.astype(int); age = meta.age_raw.values
    sub = {"overall": ok.copy(), "age65": ok_age & (age >= 65), "age75": ok_age & (age >= 75)}
    if name == "code15": sub["age65_fulllen"] = sub["age65"] & (meta.span_samples_400hz.values >= 4096)
    if name == "chapman":
        for s in ("Shaoxing", "Ningbo"): sub[f"age65_{s}"] = sub["age65"] & (meta.site_inferred.values == s)
    fac = pd.factorize(pid[ok])[0]; P = fac.max() + 1
    idx_ok = np.where(ok)[0]; pat_full = np.full(len(meta), -1); pat_full[idx_ok] = fac
    rng = np.random.default_rng(seed)
    C = rng.multinomial(P, np.ones(P) / P, size=boot).astype(np.float32)
    labels = sorted(models.label.unique()); units = {}
    for lab in labels:
        for s in sorted(models[models.label == lab].init_seed.unique()):
            ix = models[(models.label == lab) & (models.init_seed == s)].model_idx.values
            units[(lab, s)] = scores[ix].mean(axis=0)                         # ensemble of the fold models
    sens = {}
    for (lab, s), p in units.items():
        for g, m in sub.items():
            if m.sum() == 0 or y[m].sum() == 0: continue
            pt = sens_weighted(p[m], y[m], np.ones((1, m.sum()), np.float32))[0]
            bs = sens_weighted(p[m], y[m], C[:, pat_full[m]])
            sens[(lab, s, g)] = (pt, bs)
    rows = []
    for lab in labels:
        ss = sorted(s for (l, s) in units if l == lab); r = dict(arm=lab, n_seeds=len(ss))
        for g in sub:
            if (lab, ss[0], g) not in sens: continue
            pts = np.array([sens[(lab, s, g)][0] for s in ss]); bs = np.mean([sens[(lab, s, g)][1] for s in ss], axis=0)
            lo, hi = np.percentile(bs, [2.5, 97.5])
            r.update({f"sens95_{g}": pts.mean(), f"sens95_{g}_lo": lo, f"sens95_{g}_hi": hi, f"sens95_{g}_seedsd": pts.std(ddof=1) if len(pts) > 1 else np.nan})
        r["auc_overall"] = np.mean([auc(y[sub["overall"]], units[(lab, s)][sub["overall"]]) for s in ss])
        r["auc_age65"] = np.mean([auc(y[sub["age65"]], units[(lab, s)][sub["age65"]]) for s in ss])
        rows.append(r)
    arms = pd.DataFrame(rows)
    crow = []
    for a, b in PAIRS:
        if a not in labels or b not in labels: continue
        common = sorted(set(s for (l, s) in units if l == a) & set(s for (l, s) in units if l == b))
        for g in sub:
            if (a, common[0], g) not in sens: continue
            d = np.array([sens[(a, s, g)][0] - sens[(b, s, g)][0] for s in common])
            bs = np.mean([sens[(a, s, g)][1] - sens[(b, s, g)][1] for s in common], axis=0)
            lo95, hi95 = np.percentile(bs, [2.5, 97.5]); lo90, hi90 = np.percentile(bs, [5, 95])
            crow.append(dict(a=a, b=b, subgroup=g, n_seeds=len(common), diff=d.mean(), lo95=lo95, hi95=hi95, lo90=lo90, hi90=hi90,
                             seedsd=d.std(ddof=1) if len(common) > 1 else np.nan, p_boot=min(1.0, 2 * min((bs <= 0).mean(), (bs >= 0).mean()) + 1.0 / (boot + 1)),
                             equivalent_pm4=(bool(lo90 > -MARGIN and hi90 < MARGIN) if name == "code15" else None)))
    # Age-disparity quantities (outside the primary outputs): sensitivity below 65 and the age gap (65+ minus under 65), pooled and,
    # for Chapman plus Ningbo, within each inferred hospital, per arm and as contrasts between arms.
    xgroups = [("under65", "age65", "overall")] + ([(f"under65_{s}", f"age65_{s}", s) for s in ("Shaoxing", "Ningbo")] if name == "chapman" else [])
    under = ok_age & (age < 65)
    xsub = {"under65": under}
    if name == "chapman":
        for s in ("Shaoxing", "Ningbo"): xsub[f"under65_{s}"] = under & (meta.site_inferred.values == s)
    for (lab, s), p in units.items():
        for gu, g65, _ in xgroups:
            m = xsub[gu]
            pt_u = sens_weighted(p[m], y[m], np.ones((1, m.sum()), np.float32))[0]; bs_u = sens_weighted(p[m], y[m], C[:, pat_full[m]])
            sens[(lab, s, gu)] = (pt_u, bs_u)
            sens[(lab, s, gu.replace("under65", "gap"))] = (sens[(lab, s, g65)][0] - pt_u, sens[(lab, s, g65)][1] - bs_u)
    xnames = [g for gu, _, _ in xgroups for g in (gu, gu.replace("under65", "gap"))]
    xarm, xcon = [], []
    for lab in labels:
        ss = sorted(s for (l, s) in units if l == lab); r = dict(arm=lab, n_seeds=len(ss))
        for g in xnames:
            pts = np.array([sens[(lab, s, g)][0] for s in ss]); bs = np.mean([sens[(lab, s, g)][1] for s in ss], axis=0)
            lo, hi = np.percentile(bs, [2.5, 97.5]); r.update({g: pts.mean(), f"{g}_lo": lo, f"{g}_hi": hi})
        xarm.append(r)
    for a_, b_ in PAIRS:
        if a_ not in labels or b_ not in labels: continue
        common = sorted(set(s for (l, s) in units if l == a_) & set(s for (l, s) in units if l == b_))
        for g in xnames:
            d = np.array([sens[(a_, s, g)][0] - sens[(b_, s, g)][0] for s in common])
            bs = np.mean([sens[(a_, s, g)][1] - sens[(b_, s, g)][1] for s in common], axis=0)
            lo95, hi95 = np.percentile(bs, [2.5, 97.5])
            xcon.append(dict(a=a_, b=b_, quantity=g, n_seeds=len(common), diff=d.mean(), lo95=lo95, hi95=hi95, lo90=np.percentile(bs, 5), hi90=np.percentile(bs, 95),
                             p_boot=min(1.0, 2 * min((bs <= 0).mean(), (bs >= 0).mean()) + 1.0 / (boot + 1)),
                             p_signflip=signflip_p(d) if len(common) > 1 else np.nan))
    os.makedirs(os.path.join(out_dir, "agegap"), exist_ok=True)
    pd.DataFrame(xarm).to_csv(os.path.join(out_dir, "agegap", f"ext_{name}_arms.csv"), index=False)
    pd.DataFrame(xcon).to_csv(os.path.join(out_dir, "agegap", f"ext_{name}_contrasts.csv"), index=False)
    con = pd.DataFrame(crow)
    arms.to_csv(os.path.join(out_dir, f"ext_{name}_arms.csv"), index=False); con.to_csv(os.path.join(out_dir, f"ext_{name}_contrasts.csv"), index=False)
    summ = dict(cohort=name, rows_used=int(ok.sum()), units_of_resampling=int(P), resampling="patient" if name == "code15" else "record",
                positives=int(y[ok].sum()), **{f"n_{g}": int(m.sum()) for g, m in sub.items()}, **{f"pos_{g}": int(y[m].sum()) for g, m in sub.items()}, boot=boot)
    json.dump(summ, open(os.path.join(out_dir, f"ext_{name}_summary.json"), "w"), indent=2)
    pd.set_option("display.width", 220, "display.max_columns", 30)
    print(f"[{name}]", json.dumps(summ)); print(arms[["arm", "sens95_overall", "sens95_age65", "sens95_age65_lo", "sens95_age65_hi", "sens95_age75", "auc_overall"]].round(3).to_string(index=False))
    if len(con): print(con[con.subgroup.isin(["age65", "age75"])][["a", "b", "subgroup", "diff", "lo95", "hi95", "p_boot", "equivalent_pm4"]].round(3).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True); ap.add_argument("--cache-dir", required=True); ap.add_argument("--out-dir", required=True)
    ap.add_argument("--boot", type=int, default=1000); ap.add_argument("--cohorts", nargs="+", default=["code15", "chapman"])
    a = ap.parse_args()
    models = pd.read_csv(os.path.join(a.scores, "ext_models.csv"))
    for name in a.cohorts:
        meta = pd.read_csv(os.path.join(a.cache_dir, f"{name}_meta.csv"))
        sc = np.load(os.path.join(a.scores, f"ext_{name}.npy"), mmap_mode="r")
        run_cohort(name, sc, models, meta, a.boot, a.out_dir)
