"""
run_derm.py: job suites, listing and the worker for Task B (dermatology, PAD-UFES-20, smartphone eligibility).

    python3 run_derm.py count
    python3 run_derm.py worker --suite derm_primary --gpu 0 --shard 0 --nshards 8 --cache-dir ... --out-dir ...

Suites (declared in protocol/taskB_protocol.md):
    derm_primary    5 arms x 20 seeds x 10 folds        smartphone eligibility, engage=tilt, repair target = pool
    derm_frontier   repr_aware at boost 0.25 / 0.5 / 0.75, same seeds and folds
    derm_volume     recruit_random (volume control), same seeds and folds
    derm_sens_*     10 seeds x 10 folds, arms size_matched / availability / repr_aware / post_hoc:
                      none (ownership only), consent, shallow, steep (age-coefficient bounds), stream (repair target = stream)
    derm_val_common arms size_matched / availability / repr_aware / post_hoc, 20 seeds x 10 folds, checkpoint selected on the
                    complete validation fold with unit loss weights (uniform is shared with derm_primary); also
                    recruit_random and repr_aware at boost 0.25 / 0.5 / 0.75
    derm_arch_wide  5 arms x 10 seeds x 10 folds with CompactDermNet at twice the channel width (base_ch 48)
    derm_sens_band  categorical age bands instead of linear age
A job is complete when its .json exists; workers skip completed jobs.
"""
import argparse, itertools, os, sys, time, traceback
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "modelling"))

PRIMARY_CFG = dict(device="phone", engage="tilt", age_bound=None, repair_target="pool")
ARMS5 = ["uniform", "size_matched", "availability", "repr_aware", "post_hoc"]
ARMS4 = ["size_matched", "availability", "repr_aware", "post_hoc"]
FOLDS = list(range(1, 11))
SENS = {"none": dict(engage="none"), "consent": dict(engage="consent"), "shallow": dict(age_bound="shallow"),
        "steep": dict(age_bound="steep"), "stream": dict(repair_target="stream"), "band": dict(age_bound="band")}
VC_EXTRA = [("recruit_random", 1.0), ("repr_aware", 0.25), ("repr_aware", 0.5), ("repr_aware", 0.75)]
WIDE = 48


def _job(suite, arm, cfg, init_seed, elig_seed, fold, boost=1.0):
    return dict(task="derm", suite=suite, arm=arm, boost=boost, init_seed=init_seed, elig_seed=elig_seed, fold=fold, cfg=dict(cfg))


def jobs_for(suite):
    out = []
    if suite == "derm_primary":
        for s, f, a in itertools.product(range(20), FOLDS, ARMS5): out.append(_job(suite, a, PRIMARY_CFG, s, s, f))
    elif suite == "derm_frontier":
        for s, f, b in itertools.product(range(20), FOLDS, (0.25, 0.5, 0.75)): out.append(_job(suite, "repr_aware", PRIMARY_CFG, s, s, f, boost=b))
    elif suite == "derm_volume":
        for s, f in itertools.product(range(20), FOLDS): out.append(_job(suite, "recruit_random", PRIMARY_CFG, s, s, f))
    elif suite == "derm_val_common":
        for s, f, a in itertools.product(range(20), FOLDS, ARMS4): out.append(_job(suite, a, dict(PRIMARY_CFG, val_common=True), s, s, f))
        for s, f, (a, b) in itertools.product(range(20), FOLDS, VC_EXTRA): out.append(_job(suite, a, dict(PRIMARY_CFG, val_common=True), s, s, f, boost=b))
    elif suite == "derm_arch_wide":
        for s, f, a in itertools.product(range(10), FOLDS, ARMS5): out.append(_job(suite, a, dict(PRIMARY_CFG, base_ch=WIDE), s, s, f))
    elif suite.startswith("derm_sens_"):
        cfg = dict(PRIMARY_CFG, **SENS[suite[len("derm_sens_"):]])
        for s, f, a in itertools.product(range(10), FOLDS, ARMS4): out.append(_job(suite, a, cfg, s, s, f))
    else:
        raise SystemExit(f"unknown suite {suite}")
    return out


SUITES = ["derm_primary", "derm_frontier", "derm_volume", "derm_val_common", "derm_arch_wide"] + [f"derm_sens_{k}" for k in SENS]


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("count")
    w = sp.add_parser("worker")
    w.add_argument("--suite", required=True); w.add_argument("--gpu", type=int, default=0)
    w.add_argument("--shard", type=int, default=0); w.add_argument("--nshards", type=int, default=1)
    w.add_argument("--cache-dir", required=True); w.add_argument("--out-dir", required=True)
    w.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    if a.cmd == "count":
        tot = 0
        for s in SUITES:
            n = len(jobs_for(s)); tot += n; print(f"{s:14s}{n:6d}")
        print(f"{'total':14s}{tot:6d}"); return
    os.environ["CUDA_VISIBLE_DEVICES"] = str(a.gpu)
    import torch, trainer_derm as T
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
    co = T.CohortDerm.load_pad(a.cache_dir, dev)
    mine = jobs_for(a.suite)[a.shard::a.nshards]
    if a.limit: mine = mine[:a.limit]
    print(f"[derm worker gpu{a.gpu} shard {a.shard}/{a.nshards}] {len(mine)} jobs, device {dev}", flush=True)
    t0 = time.time(); done = 0
    def wait_free(gb):
        """The server is shared: wait until this GPU has `gb` GB free before starting a job."""
        if dev.type != "cuda":
            return
        while torch.cuda.mem_get_info()[0] / 2**30 < gb:
            print(f"  waiting for GPU memory (free {torch.cuda.mem_get_info()[0]/2**30:.1f} GB)", flush=True); time.sleep(60)

    def is_oom(e):
        t = str(e)
        return isinstance(e, torch.OutOfMemoryError) or "out of memory" in t or "CUBLAS_STATUS_ALLOC_FAILED" in t or "CUDA error" in t

    for j in mine:
        d, tries = None, 0
        while True:
            try:
                wait_free(3.0)
                d = T.run_and_save(co, j, a.out_dir); break
            except Exception as e:
                tries += 1
                if is_oom(e) and tries < 30:                # memory taken by another user: wait and retry the SAME job
                    print(f"  OOM on {T.job_id(j)}, retry {tries}", flush=True); torch.cuda.empty_cache(); time.sleep(60); continue
                print("JOB FAILED", T.job_id(j)); traceback.print_exc(); break
        if d is not None:
            done += 1
            print(f"  {done}/{len(mine)} {d['job_id']} steps={d['steps']} {d['seconds']}s elapsed={time.time()-t0:.0f}s", flush=True)
    print("worker finished", flush=True)


if __name__ == "__main__":
    main()
