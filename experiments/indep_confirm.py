"""
indep_confirm.py: retrain the key arms for test folds 7 to 10 with the hyperparameters selected on the selection restricted to
folds 1 to 6 (tune_independent.py). Needed only when the selection differs from the values used in the main experiments;
otherwise the main-experiment predictions for folds 7 to 10 are already free of selection on those folds.

    python3 indep_confirm.py --task ecg --selection <selection_ecg.json> --cache-dir <cache> --out-dir <results/ecg> --gpu 0 --shard 0 --nshards 4
Suite name: indep_conf (ECG) / derm_indep_conf. Arms uniform, size_matched, availability, repr_aware, recruit_random; seeds 0 to 19; test folds 7 to 10.
"""
import argparse, itertools, json, os, sys, time, traceback
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "modelling"))

ARMS = ["uniform", "size_matched", "availability", "repr_aware", "recruit_random"]
FOLDS = [7, 8, 9, 10]


def jobs_for(task):
    if task == "ecg": from run import PRIMARY_CFG; suite = "indep_conf"
    else: from run_derm import PRIMARY_CFG; suite = "derm_indep_conf"
    return [dict(task=task, suite=suite, arm=a, boost=1.0, init_seed=s, elig_seed=s, fold=f, cfg=dict(PRIMARY_CFG))
            for s, f, a in itertools.product(range(20), FOLDS, ARMS)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["ecg", "derm"], required=True); ap.add_argument("--selection", required=True)
    ap.add_argument("--cache-dir", required=True); ap.add_argument("--out-dir", required=True); ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--shard", type=int, default=0); ap.add_argument("--nshards", type=int, default=1); ap.add_argument("--count", action="store_true")
    a = ap.parse_args()
    jobs = jobs_for(a.task)
    if a.count: print(len(jobs)); return
    sel = json.load(open(a.selection))["selected"]
    os.environ["CUDA_VISIBLE_DEVICES"] = str(a.gpu)
    import torch
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if a.task == "ecg": import trainer as T; co = T.Cohort.load_ptbxl(a.cache_dir, dev)
    else: import trainer_derm as T; co = T.CohortDerm.load_pad(a.cache_dir, dev)
    tc = dict(lr=sel["lr"], epoch_equiv=sel["epoch_equiv"]); mine = jobs[a.shard::a.nshards]
    print(f"[indep_confirm {a.task} shard {a.shard}/{a.nshards}] {len(mine)} jobs, train config {tc}", flush=True)
    t0 = time.time(); done = 0
    for j in mine:
        tries = 0
        while True:
            try:
                while dev.type == "cuda" and torch.cuda.mem_get_info()[0] / 2**30 < 3.0: time.sleep(60)
                d = T.run_and_save(co, j, a.out_dir, train_cfg=tc); break
            except Exception as e:
                tries += 1; t = str(e)
                if ("out of memory" in t or "CUBLAS_STATUS_ALLOC_FAILED" in t or "CUDA error" in t) and tries < 30:
                    torch.cuda.empty_cache(); time.sleep(60); continue
                print("JOB FAILED", T.job_id(j)); traceback.print_exc(); d = None; break
        if d is not None:
            done += 1; print(f"  {done}/{len(mine)} {d['job_id']} {d['seconds']}s elapsed={time.time()-t0:.0f}s", flush=True)
    print("worker finished", flush=True)


if __name__ == "__main__":
    main()
