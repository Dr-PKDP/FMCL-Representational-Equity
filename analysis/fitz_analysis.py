"""
fitz_analysis.py: Fitzpatrick17k (verified images) scored by the PAD-trained Task B models. Descriptive.

    python3 fitz_analysis.py --scores <dir with fitz_scores.npy, fitz_models.csv> --cache-dir <cache> --out-dir <dir> [--boot 1000]

For each (arm, seed) the ten fold models are averaged. Subgroups: all images, skin type I-II, III-IV, V-VI (images with unknown
skin type are in "all" only). Sensitivity at 95% specificity (threshold set within the subgroup) and AUC. The bootstrap resamples
IMAGES (the dataset has no patient identifier), one resample shared by all arms and seeds. Contrasts as in the other analyses.
"""
import argparse, json, os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ecg_analysis import sens_weighted, auc, SECONDARY

PAIRS = [("availability", "uniform")] + SECONDARY


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True); ap.add_argument("--cache-dir", required=True); ap.add_argument("--out-dir", required=True)
    ap.add_argument("--boot", type=int, default=1000)
    a = ap.parse_args()
    models = pd.read_csv(os.path.join(a.scores, "fitz_models.csv")); sc = np.load(os.path.join(a.scores, "fitz_scores.npy"), mmap_mode="r")
    meta = pd.read_csv(os.path.join(a.cache_dir, "fitz_meta.csv")); ok = (meta.loaded == 1).values
    y = meta.has_malignant.values.astype(int); st = meta.fitzpatrick.values
    sub = {"all": ok, "I-II": ok & np.isin(st, [1, 2]), "III-IV": ok & np.isin(st, [3, 4]), "V-VI": ok & np.isin(st, [5, 6])}
    N = len(meta); rng = np.random.default_rng(20261107)
    C = rng.multinomial(N, np.ones(N) / N, size=a.boot).astype(np.float32)
    labels = sorted(models.label.unique()); units = {}
    for lab in labels:
        for s in sorted(models[models.label == lab].init_seed.unique()):
            ix = models[(models.label == lab) & (models.init_seed == s)].model_idx.values; units[(lab, s)] = sc[ix].mean(axis=0)
    sens = {}
    for (lab, s), p in units.items():
        for g, m in sub.items():
            if m.sum() == 0 or y[m].sum() == 0 or y[m].sum() == m.sum(): continue
            sens[(lab, s, g)] = (sens_weighted(p[m], y[m], np.ones((1, m.sum()), np.float32))[0], sens_weighted(p[m], y[m], C[:, m]))
    rows = []
    for lab in labels:
        ss = sorted(s for (l, s) in units if l == lab); r = dict(arm=lab, n_seeds=len(ss))
        for g in sub:
            if (lab, ss[0], g) not in sens: continue
            pts = np.array([sens[(lab, s, g)][0] for s in ss]); bs = np.mean([sens[(lab, s, g)][1] for s in ss], axis=0)
            lo, hi = np.percentile(bs, [2.5, 97.5])
            r.update({f"sens95_{g}": pts.mean(), f"sens95_{g}_lo": lo, f"sens95_{g}_hi": hi, f"auc_{g}": np.mean([auc(y[sub[g]], units[(lab, s)][sub[g]]) for s in ss])})
        rows.append(r)
    arms = pd.DataFrame(rows); crow = []
    for x, b in PAIRS:
        if x not in labels or b not in labels: continue
        common = sorted(set(s for (l, s) in units if l == x) & set(s for (l, s) in units if l == b))
        for g in sub:
            if (x, common[0], g) not in sens: continue
            d = np.array([sens[(x, s, g)][0] - sens[(b, s, g)][0] for s in common]); bs = np.mean([sens[(x, s, g)][1] - sens[(b, s, g)][1] for s in common], axis=0)
            crow.append(dict(a=x, b=b, subgroup=g, n_seeds=len(common), diff=d.mean(), lo95=np.percentile(bs, 2.5), hi95=np.percentile(bs, 97.5)))
    con = pd.DataFrame(crow)
    arms.to_csv(os.path.join(a.out_dir, "fitz_arms.csv"), index=False); con.to_csv(os.path.join(a.out_dir, "fitz_contrasts.csv"), index=False)
    summ = dict(n_used=int(ok.sum()), **{f"n_{g}": int(m.sum()) for g, m in sub.items()}, **{f"pos_{g}": int(y[m].sum()) for g, m in sub.items()}, resampling="image", boot=a.boot)
    json.dump(summ, open(os.path.join(a.out_dir, "fitz_summary.json"), "w"), indent=2)
    pd.set_option("display.width", 220, "display.max_columns", 30)
    print(json.dumps(summ)); print(arms.round(3).to_string(index=False)); print(con.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
