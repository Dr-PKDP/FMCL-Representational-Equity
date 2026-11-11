"""
seed_patient_bootstrap.py: intervals that add training-seed variation to the patient bootstrap.

    python3 seed_patient_bootstrap.py [--task ecg|derm] --results <results/ecg> --cache-dir <cache> --out-dir <analysis out> \
        --suites primary frontier volume_control val_common [--recruit-jobs recruit_stats_jobs.csv] [--kappa 2] [--boot 2000]

The intervals of ecg_analysis.py resample patients and keep the seeds fixed, so they are conditional on the trained models.
Here every replicate draws the same patient resample as ecg_analysis.py (same generator seed) and, in addition, a resample of
the seeds with replacement (shared by the two arms of a contrast, which are paired by seed). The contrast replicate is the
mean over the resampled seeds of the per-seed paired difference. With --recruit-jobs, remediation efficiency
eta = (kappa - 1) R_a / d is also bootstrapped jointly: R_a of a replicate is the mean of the per-seed recruited
participations (mean over the ten fold jobs of that seed) over the same resampled seeds, so the interval also carries the
seed-to-seed variation of R_a. The uncertainty of the HINTS coefficients is not resampled.
Outputs: <out-dir>/seedpatient/<suite>_contrasts.csv (and <suite>_eta.csv with --recruit-jobs)
"""
import argparse, os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ecg_analysis as EA

QUANTITIES = ["overall", "age65", "under65", "gap"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=list(EA.TASKS), default="ecg")
    ap.add_argument("--results", required=True); ap.add_argument("--cache-dir", required=True); ap.add_argument("--out-dir", required=True)
    ap.add_argument("--suites", nargs="+", required=True); ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--folds", type=int, nargs="+", default=None)
    ap.add_argument("--recruit-jobs", default=None); ap.add_argument("--kappa", type=float, default=2.0)
    a = ap.parse_args()
    EA.TASK.clear(); EA.TASK.update(EA.TASKS[a.task])
    meta = pd.read_csv(os.path.join(a.cache_dir, EA.TASK["meta"])); meta = meta[meta.loaded == 1].sort_values(EA.TASK["rid"]).reset_index(drop=True)
    if a.folds is not None: meta = meta[meta.strat_fold.isin(a.folds)].reset_index(drop=True)
    canon = meta[EA.TASK["rid"]].values; y = meta[EA.TASK["label"]].values.astype(int); age = meta.age_capped.values
    sub = {"overall": np.ones(len(meta), bool), "age65": age >= 65, "under65": age < 65}
    pat = pd.factorize(meta.patient_id.values)[0]; P = pat.max() + 1
    C = np.random.default_rng(20261107).multinomial(P, np.ones(P) / P, size=a.boot).astype(np.float32)   # as in ecg_analysis.py
    rj = pd.read_csv(a.recruit_jobs) if a.recruit_jobs else None
    os.makedirs(os.path.join(a.out_dir, "seedpatient"), exist_ok=True)
    for suite in a.suites:
        preds, _, kind = EA.merged_preds(a.results, suite, canon, a.folds)
        labels = sorted(preds); pairs = EA.pairs_for(kind, labels)
        if "repr_aware" in preds and "repr_aware@0.75" in preds: pairs = pairs + [("repr_aware", "repr_aware@0.75")]
        sens = {}
        for lab in labels:
            for sk, p in preds[lab].items():
                for g in ("overall", "age65", "under65"):
                    m = sub[g]
                    sens[(lab, sk, g)] = (EA.sens_weighted(p[m], y[m], np.ones((1, m.sum()), np.float32))[0],
                                          EA.sens_weighted(p[m], y[m], C[:, pat[m]]))
                sens[(lab, sk, "gap")] = tuple(sens[(lab, sk, "age65")][i] - sens[(lab, sk, "under65")][i] for i in (0, 1))
        rng = np.random.default_rng(20261108)
        rows, erows = [], []
        for x, z in pairs:
            if x not in preds or z not in preds: continue
            common = sorted(set(preds[x]) & set(preds[z])); S = len(common)
            idx = rng.integers(0, S, size=(a.boot, S))                         # seed resample per replicate, shared by both arms
            for g in QUANTITIES:
                pt = np.array([sens[(x, s, g)][0] - sens[(z, s, g)][0] for s in common])
                D = np.array([sens[(x, s, g)][1] - sens[(z, s, g)][1] for s in common])       # (S, boot)
                cond = D.mean(axis=0); joint = D[idx, np.arange(a.boot)[:, None]].mean(axis=1)
                rows.append(dict(a=x, b=z, quantity=g, n_seeds=S, diff=pt.mean(),
                                 lo95_patient=np.percentile(cond, 2.5), hi95_patient=np.percentile(cond, 97.5),
                                 lo95_joint=np.percentile(joint, 2.5), hi95_joint=np.percentile(joint, 97.5),
                                 p_joint=min(1.0, 2 * min((joint <= 0).mean(), (joint >= 0).mean()) + 1.0 / (a.boot + 1))))
                if rj is not None and g == "age65" and z == "availability" and x in set(rj.label):
                    Rs = rj[rj.label == x].groupby("seed").R_rec.mean()
                    R = np.array([Rs.loc[s[0]] for s in common])
                    d_b = 100 * joint; R_b = R[idx].mean(axis=1)
                    eta_b = np.where(d_b > 0, (a.kappa - 1) * R_b / np.where(d_b > 0, d_b, 1), np.inf)
                    lo_c, hi_c = 100 * np.percentile(cond, [2.5, 97.5])
                    erows.append(dict(arm=x, kappa=a.kappa, R_rec=R.mean(), R_rec_seed_sd=R.std(ddof=1), d_pp=100 * pt.mean(),
                                      eta=(a.kappa - 1) * R.mean() / (100 * pt.mean()) if pt.mean() > 0 else np.nan,
                                      eta_lo_patient=(a.kappa - 1) * R.mean() / hi_c if lo_c > 0 else np.nan,
                                      eta_hi_patient=(a.kappa - 1) * R.mean() / lo_c if lo_c > 0 else np.inf,
                                      eta_lo_joint=np.quantile(eta_b, 0.025, method="lower"), eta_hi_joint=np.quantile(eta_b, 0.975, method="higher")))
        out = pd.DataFrame(rows); out.to_csv(os.path.join(a.out_dir, "seedpatient", f"{suite}_contrasts.csv"), index=False)
        pd.set_option("display.width", 220, "display.max_columns", 30)
        print(f"[{suite}]"); print(out[out.quantity != "overall"].round(4).to_string(index=False))
        if erows:
            e = pd.DataFrame(erows); e.to_csv(os.path.join(a.out_dir, "seedpatient", f"{suite}_eta.csv"), index=False)
            print(e.round(1).to_string(index=False))


if __name__ == "__main__":
    main()
