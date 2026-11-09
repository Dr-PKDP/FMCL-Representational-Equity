"""
tune.py: choose learning rate and epoch-equivalent budget on the UNIFORM arm, by VALIDATION loss only.
Test-fold labels are never read here. Uses seeds 900-902 (outside the analysis seeds) and test folds 1, 5, 9.
Full budget, best-validation checkpoint. Prints a table; the chosen values are DEFAULT_TRAIN in modelling/trainer.py.
    python3 tune.py --cache-dir <cache>
"""
import argparse, itertools, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "modelling"))
import torch, trainer as T
from run import PRIMARY_CFG

ap = argparse.ArgumentParser(); ap.add_argument("--cache-dir", required=True); ap.add_argument("--gpu", type=int, default=0)
ap.add_argument("--lr", type=float, default=None, help="run one learning rate only (to split the grid over processes)")
a = ap.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(a.gpu)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
co = T.Cohort.load_ptbxl(a.cache_dir, dev)
rows = []
for lr, ee in itertools.product((3e-4, 1e-3, 3e-3) if a.lr is None else (a.lr,), (20, 40, 80)):
    v = []
    for s, f in itertools.product((900, 901, 902), (1, 5, 9)):
        job = dict(task="ecg", suite="tune", arm="uniform", boost=1.0, init_seed=s, elig_seed=s, fold=f, cfg=PRIMARY_CFG)
        _, d = T.run_job(co, job, dict(lr=lr, epoch_equiv=ee))
        v.append(d["best_val_loss"])
    rows.append((lr, ee, float(np.mean(v)))); print(f"lr={lr:g} epoch_equiv={ee:3d}  mean best val loss={np.mean(v):.4f}", flush=True)
best = min(rows, key=lambda r: r[2]); print("best in this run:", dict(lr=best[0], epoch_equiv=best[1], val=round(best[2], 4)))
