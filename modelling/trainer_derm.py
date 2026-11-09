"""
trainer_derm.py

Task B training job (PAD-UFES-20, smartphone images). Same design as trainer.py (Task A): the unit of participation is the
PATIENT, eligibility persists for the whole job, each round the coordinator selects K eligible patients with probability
proportional to their selection weight and takes one random image from each, the step budget is stated in epoch-equivalents of
the full training set and is identical for every arm, the checkpoint with the lowest validation loss is kept, and the validation
fold passes through the same eligibility rule. The test fold is never filtered.

Differences from Task A, all arm-independent: images instead of ECGs (CompactDermNet, 128 x 128), K = 128 and batch = 64
(the cohort has 1,373 patients), random flips and transposition of each training image (the dihedral group, applied on the GPU),
unknown sex enters the eligibility model as 0.5, and a missing age is replaced by the cohort median for the eligibility draw only.
"""
import hashlib, json, math, os, sys, time
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "participation"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import arms as A
from models import CompactDermNet
from trainer import _val_tensors, _evaluate_loss, job_id, val_fold_of

DEFAULT_TRAIN_DERM = dict(epoch_equiv=40.0, K=128, batch=64, lr=3e-3, lr_min=1e-5, weight_decay=1e-2,
                          eval_every_rounds=10, patience=None, amp=True, augment=True)


class PatientSetDerm:
    """Images of one set (a list of folds) grouped by patient."""
    def __init__(self, meta, rec_idx, age_fill):
        m = meta.iloc[rec_idx]
        order = np.lexsort((m.row_idx.values, m.patient_id.values))
        rec = np.asarray(rec_idx)[order]; pid = m.patient_id.values[order]
        self.pat_ids, first, self.count = np.unique(pid, return_index=True, return_counts=True)
        self.start = first; self.rec = rec
        fr = rec[first]
        age = meta.age_capped.values[fr].astype(float)
        self.age = np.where(np.isnan(age), age_fill, age)
        sex = meta.sex.values[fr]
        self.female = np.where(sex == -1, 0.5, (sex == 1).astype(float))
        self.n_patients = len(self.pat_ids)


class CohortDerm:
    def __init__(self, X, meta, device):
        meta = meta.reset_index(drop=True)
        self.meta = meta; self.device = device
        self.X = X if torch.is_tensor(X) else torch.as_tensor(np.asarray(X), dtype=torch.float32)
        self.X = self.X.to(device)
        self.y = torch.as_tensor(meta.has_malignant.values.astype(np.float32), device=device)
        self.folds = sorted(meta.strat_fold.unique())
        self.age_fill = float(np.nanmedian(meta.age_capped.values))

    @staticmethod
    def load_pad(cache_dir, device):
        meta = pd.read_csv(os.path.join(cache_dir, "pad_meta.csv"))
        X = np.load(os.path.join(cache_dir, "pad_ufes20_128.npy"), mmap_mode="r")
        keep = np.where(meta.loaded.values == 1)[0]
        return CohortDerm(torch.as_tensor(np.ascontiguousarray(X[keep])), meta.iloc[keep], device)

    def folds_of(self, folds):
        return np.where(self.meta.strat_fold.isin(folds).values)[0]


def _augment(x, gen):
    """Random horizontal flip, vertical flip and transposition per image (label-preserving)."""
    b = x.shape[0]
    m = [(torch.rand(b, device=x.device, generator=gen) < 0.5).view(b, 1, 1, 1) for _ in range(3)]
    x = torch.where(m[0], x.flip(3), x); x = torch.where(m[1], x.flip(2), x)
    return torch.where(m[2], x.transpose(2, 3), x)


def run_job(cohort, job, train_cfg=None, model_fn=CompactDermNet, verbose=False, return_state=False):
    tc = dict(DEFAULT_TRAIN_DERM, **(train_cfg or {}))
    dev = cohort.device; t0 = time.time()
    folds = cohort.folds; test_f = job["fold"]; val_f = val_fold_of(test_f, folds)
    train_f = [f for f in folds if f not in (test_f, val_f)]
    meta = cohort.meta
    tr = PatientSetDerm(meta, cohort.folds_of(train_f), cohort.age_fill); va = PatientSetDerm(meta, cohort.folds_of([val_f]), cohort.age_fill)
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
        opt, lambda s: (tc["lr_min"] / tc["lr"]) + (1 - tc["lr_min"] / tc["lr"]) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total_steps))))

    sel = torch.as_tensor(pool["select_w"], dtype=torch.float32, device=dev)
    lw = torch.as_tensor(pool["loss_w"], dtype=torch.float32, device=dev)
    start = torch.as_tensor(tr.start, device=dev); cnt = torch.as_tensor(tr.count, device=dev)
    rec_all = torch.as_tensor(tr.rec, device=dev)
    n_pool = int(pool["eligible"].sum())
    vt = _val_tensors(cohort, va, vpool)
    replace = n_pool < tc["K"]

    best, best_state, n_eval, steps, rounds, participations, log = float("inf"), None, 0, 0, 0, 0, []
    for r in range(total_rounds):
        model.train()
        pat = torch.multinomial(sel, tc["K"], replacement=replace, generator=gen)
        off = (torch.rand(tc["K"], device=dev, generator=gen) * cnt[pat]).long().clamp_max(cnt[pat] - 1)
        rec = rec_all[start[pat] + off]; w = lw[pat]; y = cohort.y[rec]
        perm = torch.randperm(tc["K"], device=dev, generator=gen)
        for b in range(steps_per_round):
            ix = perm[b * tc["batch"]:(b + 1) * tc["batch"]]
            xb = cohort.X[rec[ix]]
            if tc["augment"]: xb = _augment(xb, gen)
            with torch.autocast(device_type=dev.type if dev.type != "cpu" else "cpu", dtype=torch.bfloat16, enabled=tc["amp"]):
                logit = model(xb).squeeze(1)
            loss = (F.binary_cross_entropy_with_logits(logit.float(), y[ix], reduction="none") * w[ix]).mean()
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step(); steps += 1
        participations += tc["K"]; rounds += 1
        if (r + 1) % tc["eval_every_rounds"] == 0 or r + 1 == total_rounds:
            v = _evaluate_loss(model, cohort, vt, batch=512); n_eval += 1; log.append((rounds, steps, v))
            if verbose: print(f"    round {rounds:5d} step {steps:6d} val {v:.4f}", flush=True)
            if v < best - 1e-5:
                best = v; best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}
    if best_state is not None: model.load_state_dict(best_state)

    model.eval(); probs = []; te_t = torch.as_tensor(te_idx, device=dev)
    with torch.no_grad():
        for i in range(0, len(te_idx), 512):
            probs.append(torch.sigmoid(model(cohort.X[te_t[i:i + 512]]).squeeze(1).float()).cpu().numpy())
    m = meta.iloc[te_idx]
    pred = pd.DataFrame({"row_idx": m.row_idx.values, "patient_id": m.patient_id.values, "y_true": m.has_malignant.values,
                         "y_prob": np.concatenate(probs), "age": m.age_capped.values, "age_raw": m.age_raw.values, "sex": m.sex.values})
    diag = dict(job_id=job_id(job), job=job, val_fold=val_f, train_folds=train_f, **summary,
                n_train_patients=int(tr.n_patients), n_train_records=int(n_train_rec),
                n_pool_records=int(sum(tr.count[p] for p in np.where(pool["eligible"])[0])),
                steps=steps, step_budget=total_steps, rounds=rounds, participations=participations,
                samples_processed=steps * tc["batch"], best_val_loss=best, train_cfg=tc, seconds=round(time.time() - t0, 2),
                val_log=log, n_params=int(sum(p.numel() for p in model.parameters())))
    if return_state:
        return pred, diag, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    return pred, diag


def run_and_save(cohort, job, out_dir, train_cfg=None):
    jid = job_id(job); stem = hashlib.md5(jid.encode()).hexdigest()[:10]
    base = os.path.join(out_dir, job["suite"]); os.makedirs(base, exist_ok=True)
    pth = os.path.join(base, f"{stem}.npz"); jpth = os.path.join(base, f"{stem}.json")
    if os.path.exists(pth) and os.path.exists(jpth): return None
    pred, diag, state = run_job(cohort, job, train_cfg, return_state=True)
    torch.save(state, os.path.join(base, f"{stem}.pt") + ".tmp"); os.replace(os.path.join(base, f"{stem}.pt") + ".tmp", os.path.join(base, f"{stem}.pt"))   # weights, for the Fitzpatrick17k check
    with open(pth + ".tmp", "wb") as fh: np.savez_compressed(fh, **{c: pred[c].values for c in pred.columns})
    os.replace(pth + ".tmp", pth)
    json.dump(diag, open(jpth + ".tmp", "w"), default=lambda o: o.item() if hasattr(o, "item") else str(o)); os.replace(jpth + ".tmp", jpth)
    return diag
