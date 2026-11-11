"""
recruit_stats.py: per-job recruitment quantities for the carbon accounting (Eq. 2 to 6).

    python3 recruit_stats.py --cache-dir <cache> --out-dir <analysis dir> [--seeds 20]

Training volume is matched across arms (the same participations per round and the same number of rounds), so the
operational carbon of an arm differs from the availability arm only through the participations that recruited
devices contribute. For every (arm, seed, test fold) the script rebuilds the participation pool exactly as the trainer
does (same seed keys, same training folds) and records
    n_pool_av        patients eligible without recruitment (availability pool)
    n_pool           patients eligible in the arm
    n_rec            recruited patients (eligible in the arm, not in the availability pool)
    rec_stream_share selection-weight mass of the recruited patients (share of the stream under sampling with replacement)
    rec_draw_share   expected share of the K patients drawn per round that are recruits under the trainer's scheme, which
                     samples K patients without replacement within a round (torch.multinomial, replacement=False when
                     the pool holds at least K patients); estimated from N_DRAWS simulated rounds per job
    participations   K x rounds of the job (identical to the trainer's count)
    R_rec            expected participations by recruited patients = participations x rec_draw_share
    rec_age65        share of recruits aged 65 or over
Outputs: recruit_stats_jobs.csv and recruit_stats.csv (mean over jobs per arm label).
"""
import argparse, math, os, sys
import numpy as np
import pandas as pd
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "participation")); sys.path.insert(0, os.path.join(HERE, "..", "modelling"))
import arms as A
from trainer import PatientSet, DEFAULT_TRAIN, val_fold_of

PRIMARY_CFG = dict(device="watch", engage="tilt", age_bound=None, repair_target="pool")
LABELS = [("repr_aware@0.25", "repr_aware", 0.25), ("repr_aware@0.5", "repr_aware", 0.5), ("repr_aware@0.75", "repr_aware", 0.75),
          ("repr_aware", "repr_aware", 1.0), ("recruit_random", "recruit_random", 1.0)]


N_DRAWS = 2000


def draw_share(sel, rec, K, seed, n_draws=N_DRAWS):
    """Expected share of recruits among the K patients drawn in one round, with the trainer's sampling rule."""
    w = torch.as_tensor(np.asarray(sel), dtype=torch.float32)
    idx = torch.nonzero(w > 0).squeeze(1); w = w[idx]; r = torch.as_tensor(np.asarray(rec))[idx]
    gen = torch.Generator().manual_seed(int(seed))
    replace = len(w) < K
    draws = torch.multinomial(w.expand(n_draws, -1), K, replacement=replace, generator=gen)
    return float(r[draws].float().mean())


def job_stats(meta, arm, boost, seed, fold, cfg=PRIMARY_CFG, train=DEFAULT_TRAIN):
    folds = sorted(meta.strat_fold.unique()); val_f = val_fold_of(fold, folds)
    train_f = [f for f in folds if f not in (fold, val_f)]
    tr = PatientSet(meta, np.where(meta.strat_fold.isin(train_f).values)[0])
    c = dict(cfg, boost=boost); key = (seed, fold, 0)
    base = A.build_pool("availability", tr.age, tr.female, key, c)
    pool = A.build_pool(arm, tr.age, tr.female, key, c)
    rec = pool["eligible"] & ~base["eligible"]
    assert int(rec.sum()) == int(pool["n_recruited"]), "recruit count differs from the arm's own count"
    sel = pool["select_w"]
    steps = math.ceil(train["epoch_equiv"] * len(tr.rec) / train["batch"])
    rounds = math.ceil(steps / (train["K"] // train["batch"]))
    part = rounds * train["K"]; wshare = float(sel[rec].sum() / sel.sum())
    dshare = draw_share(sel, rec, train["K"], seed * 1009 + fold)
    return dict(n_pool_av=int(base["eligible"].sum()), n_pool=int(pool["eligible"].sum()), n_rec=int(rec.sum()),
                rec_stream_share=wshare, rec_draw_share=dshare, participations=part, R_rec=part * dshare,
                rec_age65=float((tr.age[rec] >= 65).mean()) if rec.any() else float("nan"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", required=True); ap.add_argument("--out-dir", required=True); ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--age-bound", default=None, help="participation-model variant (e.g. band); output files get the suffix _<variant>")
    ap.add_argument("--labels", nargs="+", default=None, help="subset of labels, e.g. repr_aware")
    a = ap.parse_args()
    cfg = dict(PRIMARY_CFG, age_bound=a.age_bound); sfx = f"_{a.age_bound}" if a.age_bound else ""
    labels = [t for t in LABELS if a.labels is None or t[0] in a.labels]
    meta = pd.read_csv(os.path.join(a.cache_dir, "ptbxl_meta.csv")); meta = meta[meta.loaded == 1].reset_index(drop=True)
    rows = []
    for lab, arm, b in labels:
        for s in range(a.seeds):
            for f in sorted(meta.strat_fold.unique()):
                rows.append(dict(label=lab, seed=s, fold=int(f), **job_stats(meta, arm, b, s, int(f), cfg=cfg)))
    jobs = pd.DataFrame(rows); os.makedirs(a.out_dir, exist_ok=True)
    jobs.to_csv(os.path.join(a.out_dir, f"recruit_stats{sfx}_jobs.csv"), index=False)
    agg = jobs.groupby("label", sort=False).agg(n_pool_av=("n_pool_av", "mean"), n_pool=("n_pool", "mean"), n_rec=("n_rec", "mean"),
                                                rec_stream_share=("rec_stream_share", "mean"), rec_draw_share=("rec_draw_share", "mean"), participations=("participations", "mean"),
                                                R_rec=("R_rec", "mean"), rec_age65=("rec_age65", "mean"), jobs=("seed", "size")).reset_index()
    agg.to_csv(os.path.join(a.out_dir, f"recruit_stats{sfx}.csv"), index=False)
    pd.set_option("display.width", 200); print(agg.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
