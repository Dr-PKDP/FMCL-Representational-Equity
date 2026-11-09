"""Trainer test on synthetic ECG-like data (CPU, seconds). Run: python3 tests/test_trainer.py"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "modelling"))
import numpy as np, pandas as pd, torch
import trainer as T
from sklearn.metrics import roc_auc_score


def synthetic(n_pat=900, seed=0):
    rng = np.random.default_rng(seed)
    pid = np.repeat(np.arange(n_pat), rng.integers(1, 4, n_pat))
    n = len(pid)
    age_p = np.clip(rng.normal(62, 17, n_pat), 18, 90)
    fold_p = rng.integers(1, 11, n_pat)
    prev = 0.08 + 0.003 * (age_p - 18)
    y = (rng.random(n) < prev[pid]).astype(int)
    X = rng.normal(0, 1, (n, 12, 1000)).astype(np.float32)
    for i in np.where(y == 1)[0]:                                  # a learnable pulse train in two leads
        X[i, 0, ::100] += 4.0; X[i, 1, 50::100] += 3.0
    meta = pd.DataFrame(dict(ecg_id=np.arange(n), patient_id=pid, age_raw=age_p[pid], age_capped=age_p[pid],
                             sex=(pid % 2), strat_fold=fold_p[pid], has_cd=y, loaded=1))
    return X, meta


if __name__ == "__main__":
    X, meta = synthetic()
    co = T.Cohort(X, meta, torch.device("cpu"))
    cfg = dict(device="watch", engage="tilt", age_bound=None, repair_target="pool")
    tc = dict(epoch_equiv=3.0, eval_every_rounds=5, patience=20, min_evals=20, amp=False)
    res = {}
    for arm in ("uniform", "size_matched", "availability", "repr_aware", "post_hoc"):
        job = dict(task="ecg", suite="t", arm=arm, boost=1.0, init_seed=0, elig_seed=0, fold=3, cfg=cfg)
        pred, diag = T.run_job(co, job, tc)
        te = meta[meta.strat_fold == 3]
        assert len(pred) == len(te) and set(pred.ecg_id) == set(te.ecg_id), "test fold must be predicted in full"
        assert diag["steps"] == diag["step_budget"] and diag["n_params"] == 164289
        res[arm] = (roc_auc_score(pred.y_true, pred.y_prob), diag["n_eligible"], diag["n_recruited"])
        print(arm, res[arm])
    assert res["uniform"][0] > 0.85, "model fails to learn the synthetic signal"
    assert res["size_matched"][1] == res["availability"][1] and res["repr_aware"][1] >= res["availability"][1]
    # determinism of a job on CPU
    job = dict(task="ecg", suite="t", arm="availability", boost=1.0, init_seed=1, elig_seed=1, fold=2, cfg=cfg)
    p1, _ = T.run_job(co, job, tc); p2, _ = T.run_job(co, job, tc)
    assert np.allclose(p1.y_prob, p2.y_prob), "same job, different output"
    # init seed and eligibility seed are separate sources of variance
    p3, d3 = T.run_job(co, dict(job, init_seed=2), tc); p4, d4 = T.run_job(co, dict(job, elig_seed=2), tc)
    _, d1 = T.run_job(co, job, tc)
    assert d3["n_eligible"] == d1["n_eligible"]          # init seed leaves the pool unchanged
    assert d4["n_eligible"] != d1["n_eligible"] or not (d4["n_pool_records"] == d1["n_pool_records"])   # eligibility seed changes it
    assert not np.allclose(p1.y_prob, p3.y_prob) and not np.allclose(p1.y_prob, p4.y_prob)
    print("test_trainer.py OK")
