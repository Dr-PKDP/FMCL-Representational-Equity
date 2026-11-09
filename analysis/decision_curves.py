"""
decision_curves.py: net benefit of each arm's model in the age-65+ subgroup, with paired patient-level bootstrap contrasts.

    python3 decision_curves.py --task ecg --results <results/ecg> --cache-dir <cache> --out-dir <dir> [--suite primary] [--boot 1000]

Net benefit at threshold probability t: NB(t) = TP/n - FP/n x t/(1-t), computed on the pooled out-of-fold predictions of
the subgroup, averaged over seeds. References: treat everyone, treat no one. Contrasts (availability - uniform, repr_aware - availability,
size_matched - uniform, availability - size_matched) use one patient resample shared by all arms and seeds. Predicted probabilities are used as
produced by the models (no recalibration), so the curves describe the models as deployed.
"""
import argparse, os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ecg_analysis as E

THRESH = {"ecg": np.round(np.arange(0.02, 0.301, 0.02), 3), "derm": np.round(np.arange(0.10, 0.701, 0.05), 3)}
PAIRS = [("availability", "uniform"), ("size_matched", "uniform"), ("availability", "size_matched"), ("repr_aware", "availability")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=list(E.TASKS), default="ecg"); ap.add_argument("--results", required=True); ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--out-dir", required=True); ap.add_argument("--suite", default=None); ap.add_argument("--boot", type=int, default=1000)
    a = ap.parse_args(); E.TASK.update(E.TASKS[a.task])
    suite = a.suite or ("primary" if a.task == "ecg" else "derm_primary")
    meta = pd.read_csv(os.path.join(a.cache_dir, E.TASK["meta"])); meta = meta[meta.loaded == 1].sort_values(E.TASK["rid"]).reset_index(drop=True)
    canon = meta[E.TASK["rid"]].values; y = meta[E.TASK["label"]].values.astype(np.float32)
    m = (meta.age_capped.values >= 65); yy = y[m]; n = int(m.sum()); T = THRESH[a.task]; odds = (T / (1 - T)).astype(np.float32)
    pat = pd.factorize(meta.patient_id.values)[0]; P = pat.max() + 1
    C = np.random.default_rng(20261107).multinomial(P, np.ones(P) / P, size=a.boot).astype(np.float32)[:, pat[m]]      # B x n record weights
    preds, _ = E.load_suite(a.results, suite, canon)

    def nb_point_and_boot(p):
        ind = (p[m][:, None] >= T[None, :]).astype(np.float32)                    # n x T
        tp = (ind * yy[:, None]); fp = (ind * (1 - yy)[:, None])
        pt = tp.sum(0) / n - fp.sum(0) / n * odds
        W = C.sum(1, keepdims=True)
        bt = (C @ tp) / W - (C @ fp) / W * odds
        return pt, bt

    res = {}
    for lab in sorted(preds):
        pts, bts = zip(*[nb_point_and_boot(p) for p in preds[lab].values()])
        res[lab] = (np.mean(pts, axis=0), np.mean(bts, axis=0))
    prev = yy.mean(); rows = []
    for i, t in enumerate(T):
        row = dict(threshold=t, treat_all=prev - (1 - prev) * odds[i], treat_none=0.0)
        for lab, (pt, bt) in res.items():
            lo, hi = np.percentile(bt[:, i], [2.5, 97.5]); row.update({f"nb_{lab}": pt[i], f"nb_{lab}_lo": lo, f"nb_{lab}_hi": hi})
        rows.append(row)
    curves = pd.DataFrame(rows); curves["n_boot"] = a.boot; curves.to_csv(os.path.join(a.out_dir, f"dca_{a.task}_curves.csv"), index=False)
    crow = []
    for x, b in PAIRS:
        if x not in res or b not in res: continue
        d = res[x][0] - res[b][0]; bd = res[x][1] - res[b][1]
        for i, t in enumerate(T):
            crow.append(dict(a=x, b=b, threshold=t, diff=d[i], lo95=np.percentile(bd[:, i], 2.5), hi95=np.percentile(bd[:, i], 97.5)))
    con = pd.DataFrame(crow); con["n_boot"] = a.boot; con.to_csv(os.path.join(a.out_dir, f"dca_{a.task}_contrasts.csv"), index=False)
    pd.set_option("display.width", 220)
    print(f"[{a.task}] age 65+ n={n} prevalence={prev:.3f} patients={len(np.unique(pat[m]))}")
    sel = curves[curves.threshold.isin(T[::max(1, len(T) // 6)])]
    print(sel[["threshold", "treat_all"] + [f"nb_{l}" for l in sorted(res)]].round(4).to_string(index=False))
    print(con[con.threshold.isin(T[::max(1, len(T) // 6)])].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
