"""
tune_independent.py: learning rate and training budget chosen with the selection restricted to folds 1 to 6.

The original grids (tune.py, tune_derm.py) used test folds 1, 5, 9 with validation folds 2, 6, 10 and training on the rest, so
every record of the development cohort influenced some tuning job. Here the selection uses folds 1 to 6 only: the cohort is
restricted to those folds, tuning jobs use test folds 1, 3, 5 (validation folds 2, 4, 6, training on the remaining four folds
of the restricted cohort), and folds 7 to 10 are never read. Test folds 7 to 10 are then not used for hyperparameter selection (they are used for model fitting and checkpoint selection
in the other folds' models). Uniform arm, validation loss only, seeds 900 to 902, full budget, best-validation checkpoint.
The grid extends the original one by one step on each side of the selected values (learning rate and budget).

    python3 tune_independent.py --task ecg  --cache-dir <cache> --gpu 0 --out-dir <dir> [--lr 3e-3]     # one learning rate per process
    python3 tune_independent.py --task derm --cache-dir <cache> --gpu 1 --out-dir <dir>
    python3 tune_independent.py --task ecg  --out-dir <dir> --summarise                                  # merge the grid, write the selection
Outputs: <out-dir>/tuning_<task>_lr<lr>.csv per learning rate; with --summarise <out-dir>/tuning_<task>_independent.csv and
<out-dir>/selection_<task>.json (selected values, the values used in the main experiments, and whether they coincide).
"""
import argparse, glob, itertools, json, os, sys
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "modelling"))

TUNING_FOLDS = [1, 2, 3, 4, 5, 6]
TEST_FOLDS = (1, 3, 5)
SEEDS = (900, 901, 902)
GRID = {"ecg": dict(lr=(3e-4, 1e-3, 3e-3, 1e-2), ee=(10, 20, 40, 80)),
        "derm": dict(lr=(1e-3, 3e-3, 1e-2), ee=(10, 20, 40, 80))}


def restrict(co, cls, folds):
    """Cohort holding only the records of `folds`; the other records are not kept."""
    import torch
    idx = co.folds_of(folds)
    return cls(co.X[torch.as_tensor(idx, device=co.device)], co.meta.iloc[idx], co.device)


def summarise(task, out_dir):
    fs = sorted(glob.glob(os.path.join(out_dir, f"tuning_{task}_lr*.csv")))
    if not fs: raise SystemExit("no grid files")
    df = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True).sort_values(["lr", "epoch_equiv"])
    df.to_csv(os.path.join(out_dir, f"tuning_{task}_independent.csv"), index=False)
    best = df.loc[df.mean_best_val_loss.idxmin()]
    if task == "ecg":
        import trainer as T; default = dict(lr=T.DEFAULT_TRAIN["lr"], epoch_equiv=T.DEFAULT_TRAIN["epoch_equiv"])
    else:
        import trainer_derm as T; default = dict(lr=T.DEFAULT_TRAIN_DERM["lr"], epoch_equiv=T.DEFAULT_TRAIN_DERM["epoch_equiv"])
    sel = dict(task=task, selected=dict(lr=float(best.lr), epoch_equiv=float(best.epoch_equiv)), selected_val_loss=float(best.mean_best_val_loss),
               main_experiments=default, same_as_main=bool(abs(best.lr - default["lr"]) < 1e-12 and abs(best.epoch_equiv - default["epoch_equiv"]) < 1e-9),
               val_loss_at_main_values=float(df[(abs(df.lr - default["lr"]) < 1e-12) & (abs(df.epoch_equiv - default["epoch_equiv"]) < 1e-9)].mean_best_val_loss.iloc[0]),
               grid_range=float(df.mean_best_val_loss.max() - df.mean_best_val_loss.min()), n_cells=int(len(df)),
               tuning_folds=TUNING_FOLDS, tuning_test_folds=list(TEST_FOLDS), seeds=list(SEEDS))
    json.dump(sel, open(os.path.join(out_dir, f"selection_{task}.json"), "w"), indent=2)
    print(df.round(4).to_string(index=False)); print(json.dumps(sel, indent=2))
    print("SAME" if sel["same_as_main"] else "DIFFERENT")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=list(GRID), required=True); ap.add_argument("--cache-dir"); ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--out-dir", required=True); ap.add_argument("--lr", type=float, default=None); ap.add_argument("--summarise", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    if a.summarise:
        summarise(a.task, a.out_dir); return
    os.environ["CUDA_VISIBLE_DEVICES"] = str(a.gpu)
    import torch
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if a.task == "ecg":
        import trainer as T; from run import PRIMARY_CFG
        co = T.Cohort.load_ptbxl(a.cache_dir, dev); co = restrict(co, T.Cohort, TUNING_FOLDS)
    else:
        import trainer_derm as T; from run_derm import PRIMARY_CFG
        co = T.CohortDerm.load_pad(a.cache_dir, dev); co = restrict(co, T.CohortDerm, TUNING_FOLDS)
    assert sorted(co.folds) == TUNING_FOLDS and set(co.meta.strat_fold) == set(TUNING_FOLDS), "tuning cohort must hold folds 1-6 only"
    lrs = GRID[a.task]["lr"] if a.lr is None else (a.lr,)
    for lr in lrs:
        if os.path.exists(os.path.join(a.out_dir, f"tuning_{a.task}_lr{lr:g}.csv")): continue            # finished earlier
        rows = []
        for ee in GRID[a.task]["ee"]:
            v = []
            for s, f in itertools.product(SEEDS, TEST_FOLDS):
                job = dict(task=a.task, suite="tune_independent", arm="uniform", boost=1.0, init_seed=s, elig_seed=s, fold=f, cfg=PRIMARY_CFG)
                _, d = T.run_job(co, job, dict(lr=lr, epoch_equiv=ee)); v.append(d["best_val_loss"])
            rows.append(dict(lr=lr, epoch_equiv=ee, mean_best_val_loss=float(np.mean(v)), n_jobs=len(v)))
            print(f"{a.task} lr={lr:g} epoch_equiv={ee:3d}  mean best val loss={np.mean(v):.4f}", flush=True)
        pd.DataFrame(rows).to_csv(os.path.join(a.out_dir, f"tuning_{a.task}_lr{lr:g}.csv"), index=False)


if __name__ == "__main__":
    main()
