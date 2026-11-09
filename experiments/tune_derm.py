"""
tune_derm.py: choose learning rate and epoch-equivalent budget for Task B on the UNIFORM arm, by VALIDATION loss only.
Test-fold labels are never read. Seeds 900-902 (outside the analysis seeds), test folds 1, 5, 9. Full budget, best-validation checkpoint.
    python3 tune_derm.py --cache-dir <cache> --gpu 1
"""
import argparse, itertools, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "modelling"))
import torch, trainer_derm as T
from run_derm import PRIMARY_CFG

ap = argparse.ArgumentParser(); ap.add_argument("--cache-dir", required=True); ap.add_argument("--gpu", type=int, default=0)
a = ap.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(a.gpu)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
co = T.CohortDerm.load_pad(a.cache_dir, dev)
rows = []
for lr, ee in itertools.product((1e-3, 3e-3), (20, 40, 80)):
    v = []
    for s, f in itertools.product((900, 901, 902), (1, 5, 9)):
        job = dict(task="derm", suite="tune", arm="uniform", boost=1.0, init_seed=s, elig_seed=s, fold=f, cfg=PRIMARY_CFG)
        _, d = T.run_job(co, job, dict(lr=lr, epoch_equiv=ee)); v.append(d["best_val_loss"])
    rows.append((lr, ee, float(np.mean(v)))); print(f"lr={lr:g} epoch_equiv={ee:3d}  mean best val loss={np.mean(v):.4f}", flush=True)
best = min(rows, key=lambda r: r[2]); print("best:", dict(lr=best[0], epoch_equiv=best[1], val=round(best[2], 4)))
