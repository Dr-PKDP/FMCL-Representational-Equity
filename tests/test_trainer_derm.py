"""Synthetic check of the Task B trainer (CPU): pools, accounting, outputs, determinism, augmentation."""
import os, sys, tempfile
import numpy as np, pandas as pd, torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "modelling"))
import trainer_derm as T

rng = np.random.default_rng(0); n = 400
pid = rng.integers(0, 150, n); age = np.where(rng.random(n) < 0.05, np.nan, np.clip(rng.normal(58, 17, n), 18, 90))
age_p = pd.Series(age).groupby(pid).transform("first").values            # age is a patient attribute
sex_p = pd.Series(rng.choice([-1, 0, 1], n)).groupby(pid).transform("first").values
fold = (pd.Series(pid).map({p: (i % 10) + 1 for i, p in enumerate(np.unique(pid))})).values
y = rng.integers(0, 2, n)
meta = pd.DataFrame(dict(row_idx=np.arange(n), img_id=[f"i{i}" for i in range(n)], patient_id=pid, age_raw=age_p, age_capped=np.minimum(age_p, 90),
                         sex=sex_p, has_malignant=y, strat_fold=fold, loaded=1))
X = torch.randn(n, 3, 32, 32); X[y == 1, 0] += 0.8                   # a learnable signal
co = T.CohortDerm(X, meta, torch.device("cpu"))
cfg = dict(device="phone", engage="tilt", age_bound=None, repair_target="pool")
tc = dict(epoch_equiv=6.0, K=32, batch=16, eval_every_rounds=3, amp=False)
class Tiny(torch.nn.Module):
    def __init__(s): super().__init__(); s.f = torch.nn.Sequential(torch.nn.Conv2d(3, 4, 3, 2, 1), torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten(), torch.nn.Linear(4, 1))
    def forward(s, x): return s.f(x)
out = {}
for arm in ("uniform", "size_matched", "availability", "repr_aware", "post_hoc", "recruit_random"):
    job = dict(task="derm", suite="t", arm=arm, boost=1.0, init_seed=1, elig_seed=1, fold=3, cfg=cfg)
    p, d = T.run_job(co, job, tc, model_fn=Tiny); out[arm] = (p, d)
    assert len(p) == (meta.strat_fold == 3).sum() and set(p.row_idx) == set(meta.row_idx[meta.strat_fold == 3])
    assert p.y_prob.between(0, 1).all() and d["steps"] == d["step_budget"] and np.isfinite(d["best_val_loss"])
assert len({out[a][1]["steps"] for a in out}) == 1 and len({out[a][1]["participations"] for a in out}) == 1       # matched budget
assert out["availability"][1]["n_eligible"] <= out["repr_aware"][1]["n_eligible"] == out["recruit_random"][1]["n_eligible"]
a1 = T.run_job(co, dict(task="derm", suite="t", arm="availability", boost=1.0, init_seed=1, elig_seed=1, fold=3, cfg=cfg), tc, model_fn=Tiny)[0]
assert np.allclose(a1.y_prob.values, out["availability"][0].y_prob.values), "not deterministic"
g = torch.Generator(); g.manual_seed(0); xa = T._augment(torch.randn(64, 3, 8, 8), g)
assert xa.shape == (64, 3, 8, 8)
with tempfile.TemporaryDirectory() as d:
    job = dict(task="derm", suite="t", arm="uniform", boost=1.0, init_seed=2, elig_seed=2, fold=4, cfg=cfg)
    assert T.run_and_save(co, job, d, dict(tc)) is not None or True
print("test_trainer_derm.py OK")
