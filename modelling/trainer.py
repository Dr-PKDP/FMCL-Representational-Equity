"""
trainer.py

One training job = one (arm, init seed, eligibility seed, test fold, cfg). Everything the job needs sits on the
GPU: the cache (about 1 GB for PTB-XL) is loaded once per worker process, and batches are gathered by index.
No DataLoader workers.

Unit of participation is the PATIENT. Eligibility (participation/arms.py) is drawn per patient and persists
for the whole job. Each round the coordinator selects K eligible patients with probability proportional to
their selection weight and takes one random record from each. A round is followed by K / batch SGD steps.

Budget. The step budget is stated in epoch-equivalents of the full training set and is identical for every
arm, so arms differ in who is in the pool and not in how much computation they receive. Every job runs the whole
budget (cosine schedule to its end) and the checkpoint with the lowest validation loss is kept. The validation
fold passes through the SAME eligibility rule (the validation data an arm could not have collected are not
available to it). Optional early stopping (patience) exists but is off in the protocol (protocol/analysis_plan.md, addendum 1).

Evaluation. The model is applied to every record of the held-out test fold, never filtered by eligibility.
"""
import hashlib, json, math, os, sys, time
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "participation"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import arms as A
from models import CompactECGNet

DEFAULT_TRAIN = dict(epoch_equiv=20.0, K=512, batch=128, lr=3e-3, lr_min=1e-5, weight_decay=1e-2,
                     eval_every_rounds=25, patience=None, min_evals=8, amp=True)   # patience=None: full budget, best-validation checkpoint


# --------------------------------------------------------------------------- data
class PatientSet:
    """Records of one set (a list of folds) grouped by patient."""
    def __init__(self, meta, rec_idx):
        m = meta.iloc[rec_idx]
        order = np.lexsort((m.ecg_id.values, m.patient_id.values))
        rec = np.asarray(rec_idx)[order]; pid = m.patient_id.values[order]
        self.pat_ids, first, self.count = np.unique(pid, return_index=True, return_counts=True)
        self.start = first
        self.rec = rec                                           # record row indices, grouped by patient
        first_rec = rec[first]                                   # patient age/sex taken from the first record
        self.age = meta.age_capped.values[first_rec].astype(float)
        self.female = (meta.sex.values[first_rec] == 1).astype(float)
        self.n_patients = len(self.pat_ids)


class Cohort:
    """GPU-resident cohort: signals, labels and the metadata of each record."""
    def __init__(self, X, meta, device):
        meta = meta.reset_index(drop=True)
        self.meta = meta
        self.device = device
        self.X = X if torch.is_tensor(X) else torch.as_tensor(np.asarray(X), dtype=torch.float32)
        self.X = self.X.to(device)
        self.y = torch.as_tensor(meta.has_cd.values.astype(np.float32), device=device)
        self.folds = sorted(meta.strat_fold.unique())

    @staticmethod
    def load_ptbxl(cache_dir, device):
        meta = pd.read_csv(os.path.join(cache_dir, "ptbxl_meta.csv"))
        X = np.load(os.path.join(cache_dir, "ptbxl_100hz_signals.npy"), mmap_mode="r")
        keep = np.where(meta.loaded.values == 1)[0]              # unreadable rows are dropped
        return Cohort(torch.as_tensor(np.ascontiguousarray(X[keep])), meta.iloc[keep], device)

    def folds_of(self, folds):
        return np.where(self.meta.strat_fold.isin(folds).values)[0]


# --------------------------------------------------------------------------- jobs
def job_id(job):
    c = job["cfg"]
    return (f"{job['task']}__{job['suite']}__{job['arm']}__b{job.get('boost', 1.0):g}__{c['device']}-{c['engage']}-"
            f"{c.get('age_bound') or 'pt'}-{c.get('repair_target', 'na')}__i{job['init_seed']}_e{job['elig_seed']}_f{job['fold']}")


def val_fold_of(test_fold, folds):
    return folds[(folds.index(test_fold) + 1) % len(folds)]      # the next fold, cyclically


def _val_tensors(cohort, ps, pool):
    """Eligible validation records and their loss weights, built once per job."""
    pat_ok = np.where(pool["eligible"])[0]
    if len(pat_ok) == 0:
        return None
    rec = np.concatenate([ps.rec[ps.start[p]:ps.start[p] + ps.count[p]] for p in pat_ok])
    w = np.concatenate([np.full(ps.count[p], pool["loss_w"][p]) for p in pat_ok])
    return (torch.as_tensor(rec, device=cohort.device), torch.as_tensor(w, dtype=torch.float32, device=cohort.device))


def _evaluate_loss(model, cohort, vt, batch=2048):
    """Mean (loss-weighted) BCE over the eligible validation records."""
    if vt is None:
        return float("inf")
    rec_t, w_t = vt
    model.eval(); tot = 0.0
    with torch.no_grad():
        for i in range(0, len(rec_t), batch):
            r = rec_t[i:i + batch]
            logit = model(cohort.X[r]).squeeze(1).float()
            tot += float((F.binary_cross_entropy_with_logits(logit, cohort.y[r], reduction="none") * w_t[i:i + batch]).sum())
    return tot / float(w_t.sum())


def run_job(cohort, job, train_cfg=None, model_fn=CompactECGNet, verbose=False, return_state=False):
    """Returns (predictions DataFrame, diagnostics dict); with return_state also the best-checkpoint state dict (CPU)."""
    tc = dict(DEFAULT_TRAIN, **(train_cfg or {}))
    dev = cohort.device
    t0 = time.time()
    folds = cohort.folds
    test_f = job["fold"]; val_f = val_fold_of(test_f, folds)
    train_f = [f for f in folds if f not in (test_f, val_f)]
    meta = cohort.meta
    tr = PatientSet(meta, cohort.folds_of(train_f)); va = PatientSet(meta, cohort.folds_of([val_f]))
    te_idx = cohort.folds_of([test_f])

    cfg = dict(job["cfg"], boost=job.get("boost", 1.0))
    pool = A.build_pool(job["arm"], tr.age, tr.female, (job["elig_seed"], test_f, 0), cfg)
    vpool = A.build_pool(job["arm"], va.age, va.female, (job["elig_seed"], test_f, 1), cfg)
    summary = A.pool_summary(pool, tr.age)
    n_train_rec = len(tr.rec)

    torch.manual_seed(job["init_seed"] * 1000003 + test_f)
    gen = torch.Generator(device=dev); gen.manual_seed(job["init_seed"] * 7919 + test_f)
    model = model_fn().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=tc["lr"], weight_decay=tc["weight_decay"])

    steps_per_round = tc["K"] // tc["batch"]
    total_steps = int(math.ceil(tc["epoch_equiv"] * n_train_rec / tc["batch"]))
    total_rounds = int(math.ceil(total_steps / steps_per_round))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (tc["lr_min"] / tc["lr"]) + (1 - tc["lr_min"] / tc["lr"]) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / (total_steps)))))

    sel = torch.as_tensor(pool["select_w"], dtype=torch.float32, device=dev)
    lw = torch.as_tensor(pool["loss_w"], dtype=torch.float32, device=dev)
    start = torch.as_tensor(tr.start, device=dev); cnt = torch.as_tensor(tr.count, device=dev)
    rec_all = torch.as_tensor(tr.rec, device=dev)
    n_pool = int(pool["eligible"].sum())
    vt = _val_tensors(cohort, va, vpool)
    replace = n_pool < tc["K"]

    best, best_state, bad, n_eval, steps, rounds = float("inf"), None, 0, 0, 0, 0
    participations = 0; log = []
    for r in range(total_rounds):
        model.train()
        pat = torch.multinomial(sel, tc["K"], replacement=replace, generator=gen)           # K participating patients
        off = (torch.rand(tc["K"], device=dev, generator=gen) * cnt[pat]).long().clamp_max(cnt[pat] - 1)
        rec = rec_all[start[pat] + off]; w = lw[pat]; y = cohort.y[rec]
        perm = torch.randperm(tc["K"], device=dev, generator=gen)
        for b in range(steps_per_round):
            ix = perm[b * tc["batch"]:(b + 1) * tc["batch"]]
            with torch.autocast(device_type=dev.type if dev.type != "cpu" else "cpu", dtype=torch.bfloat16, enabled=tc["amp"]):
                logit = model(cohort.X[rec[ix]]).squeeze(1)
            loss = (F.binary_cross_entropy_with_logits(logit.float(), y[ix], reduction="none") * w[ix]).mean()
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step(); steps += 1
        participations += tc["K"]; rounds += 1
        if (r + 1) % tc["eval_every_rounds"] == 0 or r + 1 == total_rounds:
            v = _evaluate_loss(model, cohort, vt); n_eval += 1
            log.append((rounds, steps, v))
            if verbose:
                print(f"    round {rounds:5d} step {steps:6d} val {v:.4f}", flush=True)
            if v < best - 1e-5:
                best, bad = v, 0
                best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}
            else:
                bad += 1
                if tc["patience"] is not None and n_eval >= tc["min_evals"] and bad >= tc["patience"]:
                    break
    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval(); probs = []
    te_t = torch.as_tensor(te_idx, device=dev)
    with torch.no_grad():
        for i in range(0, len(te_idx), 2048):
            probs.append(torch.sigmoid(model(cohort.X[te_t[i:i + 2048]]).squeeze(1).float()).cpu().numpy())
    m = meta.iloc[te_idx]
    pred = pd.DataFrame({"ecg_id": m.ecg_id.values, "patient_id": m.patient_id.values, "y_true": m.has_cd.values,
                         "y_prob": np.concatenate(probs), "age": m.age_capped.values,
                         "age_raw": m.age_raw.values, "sex": m.sex.values})
    diag = dict(job_id=job_id(job), job=job, val_fold=val_f, train_folds=train_f, **summary,
                n_train_patients=int(tr.n_patients), n_train_records=int(n_train_rec),
                n_pool_records=int(sum(tr.count[p] for p in np.where(pool["eligible"])[0])),
                steps=steps, step_budget=total_steps, rounds=rounds, participations=participations,
                samples_processed=steps * tc["batch"], best_val_loss=best, stopped_early=bool(steps < total_steps),
                train_cfg=tc, seconds=round(time.time() - t0, 2), val_log=log,
                n_params=int(sum(p.numel() for p in model.parameters())))
    if return_state:
        return pred, diag, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    return pred, diag


def run_and_save(cohort, job, out_dir, train_cfg=None):
    jid = job_id(job)
    stem = hashlib.md5(jid.encode()).hexdigest()[:10]
    base = os.path.join(out_dir, f"{job['suite']}")
    os.makedirs(base, exist_ok=True)
    pth = os.path.join(base, f"{stem}.npz"); jpth = os.path.join(base, f"{stem}.json")
    if os.path.exists(pth) and os.path.exists(jpth):
        return None
    pred, diag, state = run_job(cohort, job, train_cfg, return_state=True)
    torch.save(state, os.path.join(base, f"{stem}.pt") + ".tmp"); os.replace(os.path.join(base, f"{stem}.pt") + ".tmp", os.path.join(base, f"{stem}.pt"))   # weights, for scoring external cohorts
    with open(pth + ".tmp", "wb") as fh:                       # no extra dependency: compressed numpy archive
        np.savez_compressed(fh, **{c: pred[c].values for c in pred.columns})
    os.replace(pth + ".tmp", pth)
    json.dump(diag, open(jpth + ".tmp", "w"), default=lambda o: o.item() if hasattr(o, "item") else str(o)); os.replace(jpth + ".tmp", jpth)     # json last: its presence marks completion
    return diag
