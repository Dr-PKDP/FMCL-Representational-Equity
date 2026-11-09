"""
varcomp_analysis.py: variance components of the primary endpoint (sensitivity at 95% specificity, age 65+) over a 4 x 4 grid of
initialisation seeds and eligibility seeds (folds 2, 5, 9; arms availability and repair).

    python3 varcomp_analysis.py --results <results/ecg/varcomp> --out-dir <dir>

For every (arm, init seed, eligibility seed) the predictions of the three test folds are pooled and the metric is computed
(threshold set within the 65+ subgroup). A two-way random-effects decomposition without replication (balanced 4 x 4) gives
the initialisation, eligibility and residual components; the same decomposition is applied to the paired difference
repair - availability.
"""
import argparse, glob, json, os
import numpy as np
import pandas as pd


def sens95(y, p):
    neg = np.sort(p[y == 0]); thr = neg[int(np.ceil(0.95 * len(neg))) - 1]
    return float((p[y == 1] > thr).mean())


def decompose(M):
    a, b = M.shape; g = M.mean()
    ms_i = b * ((M.mean(1) - g) ** 2).sum() / (a - 1)
    ms_e = a * ((M.mean(0) - g) ** 2).sum() / (b - 1)
    res = M - M.mean(1, keepdims=True) - M.mean(0, keepdims=True) + g
    ms_r = (res ** 2).sum() / ((a - 1) * (b - 1))
    v_i, v_e = max(0.0, (ms_i - ms_r) / b), max(0.0, (ms_e - ms_r) / a)
    return dict(var_init=v_i, var_elig=v_e, var_resid=ms_r, sd_total=float(M.std(ddof=1)), mean=float(g))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--results", required=True); ap.add_argument("--out-dir", required=True)
    a = ap.parse_args(); rows = {}
    for jp in glob.glob(os.path.join(a.results, "*.json")):
        d = json.load(open(jp)); z = np.load(jp[:-5] + ".npz"); j = d["job"]
        rows.setdefault((j["arm"], j["init_seed"], j["elig_seed"]), []).append((z["y_true"], z["y_prob"], z["age"]))
    cells = {}
    for (arm, i, e), parts in rows.items():
        y = np.concatenate([p[0] for p in parts]); p = np.concatenate([p[1] for p in parts]); age = np.concatenate([p[2] for p in parts])
        m = age >= 65; cells[(arm, i, e)] = sens95(y[m], p[m])
    seeds = sorted({k[1] for k in cells}); out = []
    mats = {}
    for arm in ("availability", "repr_aware"):
        mats[arm] = np.array([[cells[(arm, i, e)] for e in seeds] for i in seeds]) * 100
    mats["repair_minus_availability"] = mats["repr_aware"] - mats["availability"]
    for k, M in mats.items():
        r = decompose(M); r["quantity"] = k; out.append(r)
    df = pd.DataFrame(out)[["quantity", "mean", "sd_total", "var_init", "var_elig", "var_resid"]]
    df.to_csv(os.path.join(a.out_dir, "varcomp_components.csv"), index=False)
    print(df.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
