"""Tests of the hyperparameter selection restricted to folds 1 to 6, the band_vc suite, the fold-restricted analysis and the common-threshold estimand
(synthetic data, CPU, under a minute). Run: python3 tests/test_round7.py"""
import json, math, os, sys, tempfile
H = os.path.dirname(os.path.abspath(__file__))
for d in ("modelling", "analysis", "experiments", "participation"): sys.path.insert(0, os.path.join(H, "..", d))
import numpy as np, pandas as pd, torch
import trainer as T, ecg_analysis as EA, common_threshold as CT, run as R, tune_independent as TI
sys.path.insert(0, H); from test_trainer import synthetic

# 1. band_vc suite
jobs = R.jobs_for("band_vc")
assert len(jobs) == 10 * 10 * (4 + 4) and len({T.job_id(j) for j in jobs}) == len(jobs)
assert all(j["cfg"]["age_bound"] == "band" and j["cfg"]["val_common"] for j in jobs)
assert {j["arm"] for j in jobs} == {"size_matched", "availability", "repr_aware", "post_hoc", "recruit_random"} and "uniform" not in {j["arm"] for j in jobs}
assert "band_vc" in R.SUITES
print("band_vc suite ok:", len(jobs), "jobs")

# 2. the folds 1-6 selection cohort never contains folds 7-10, and tuning jobs predict only folds 1-6
X, meta = synthetic(); co = T.Cohort(X, meta, torch.device("cpu"))
sub = TI.restrict(co, T.Cohort, TI.TUNING_FOLDS)
assert sorted(sub.folds) == TI.TUNING_FOLDS and sub.meta.strat_fold.max() <= 6 and len(sub.meta) == int(meta.strat_fold.le(6).sum())
assert len(sub.X) == len(sub.meta)
job = dict(task="ecg", suite="t", arm="uniform", boost=1.0, init_seed=900, elig_seed=900, fold=5, cfg=R.PRIMARY_CFG)
pred, diag = T.run_job(sub, job, dict(epoch_equiv=1.0, eval_every_rounds=5, amp=False))
assert diag["val_fold"] == 6 and set(diag["train_folds"]) == {1, 2, 3, 4} and set(pred.ecg_id) <= set(meta[meta.strat_fold <= 6].ecg_id)
job["fold"] = 6; _, d6 = T.run_job(sub, job, dict(epoch_equiv=1.0, eval_every_rounds=5, amp=False))
assert d6["val_fold"] == 1                                     # cyclic within the restricted cohort
import trainer_derm as TD
rg = np.random.default_rng(3); nd = 300; pid = rg.integers(0, 120, nd); fo = pd.Series(pid).map({q: (i % 10) + 1 for i, q in enumerate(np.unique(pid))}).values
md = pd.DataFrame(dict(row_idx=np.arange(nd), patient_id=pid, age_raw=50.0, age_capped=50.0, sex=0, has_malignant=rg.integers(0, 2, nd), strat_fold=fo, loaded=1))
cd = TD.CohortDerm(torch.randn(nd, 3, 16, 16), md, torch.device("cpu")); sd = TI.restrict(cd, TD.CohortDerm, TI.TUNING_FOLDS)
assert sorted(sd.folds) == TI.TUNING_FOLDS and sd.meta.strat_fold.max() <= 6 and len(sd.X) == len(sd.meta)
print("folds 1-6 selection ok")

# 3. fold-restricted loading
with tempfile.TemporaryDirectory() as tmp:
    os.makedirs(os.path.join(tmp, "s"))
    for f in (3, 8):
        ids = meta[meta.strat_fold == f].ecg_id.values
        j = dict(arm="availability", boost=1.0, init_seed=0, elig_seed=0, fold=f)
        np.savez(os.path.join(tmp, "s", f"j{f}.npz"), ecg_id=ids, y_prob=np.full(len(ids), float(f)))
        json.dump(dict(job=j, n_eligible=5, n_recruited=0, share65=0.2, stream65=0.2, steps=1, participations=1, samples_processed=1, seconds=1, best_val_loss=0.1),
                  open(os.path.join(tmp, "s", f"j{f}.json"), "w"))
    canon = np.sort(meta[meta.strat_fold >= 7].ecg_id.values)
    preds, dg = EA.load_suite(tmp, "s", canon, folds=[7, 8, 9, 10])
    arr = preds["availability"][(0, 0)]
    assert list(dg.fold) == [8] and np.isfinite(arr[np.isin(canon, meta[meta.strat_fold == 8].ecg_id)]).all()
    assert np.isnan(arr[np.isin(canon, meta[meta.strat_fold.isin([7, 9, 10])].ecg_id)]).all()
print("fold-restricted loading ok")

# 4. common-threshold estimand against a direct computation, and agreement with the overall estimate of ecg_analysis
rng = np.random.default_rng(1); n = 4000
age = rng.uniform(20, 90, n); y = (rng.random(n) < 0.2 + 0.004 * (age - 20)).astype(int)
sc = rng.normal(size=n) + 1.2 * y - 0.01 * (age - 50) * y
m65 = age >= 65; masks = {"overall": np.ones(n, bool), "age65": m65, "under65": ~m65}
s, c = CT.common_threshold(sc, y, np.ones((1, n), np.float32), masks)
neg = np.sort(sc[y == 0]); t = neg[int(math.ceil(0.95 * len(neg))) - 1]; call = sc > t
for k, m in masks.items():
    assert abs(s[k][0] - (call & (y == 1) & m).sum() / ((y == 1) & m).sum()) < 1e-5, k
    assert abs(c[k][0] - (1 - (call & (y == 0) & m).sum() / ((y == 0) & m).sum())) < 1e-5, k
assert abs(s["overall"][0] - EA.sens_weighted(sc, y, np.ones((1, n), np.float32))[0]) < 1e-6
w2, _ = CT.common_threshold(sc, y, np.vstack([np.ones(n), 2 * np.ones(n)]).astype(np.float32), masks)
assert abs(w2["age65"][0] - w2["age65"][1]) < 1e-6              # doubling all weights changes nothing
print("common threshold ok")
print("all round-7 tests passed")
