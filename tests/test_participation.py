"""Invariants of propensity.py and arms.py. Run: python3 test_participation.py (no data needed)."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "participation"))
import numpy as np
import arms as A, propensity as P

rng = np.random.default_rng(0)
N = 20000
age = np.minimum(np.clip(rng.normal(60, 18, N), 18, 95), 90); fem = rng.integers(0, 2, N).astype(float)

# propensity direction: device use falls with age, engagement rises with age
for dev in ("phone", "watch"):
    assert P.prob(dev, 30, 0) > P.prob(dev, 80, 0), dev
assert P.prob("shared", 30, 0) < P.prob("shared", 80, 0)
# age is capped at 90
assert abs(P.prob("watch", 90, 0) - P.prob("watch", 300, 0)) < 1e-12
# shallow bound is flatter than steep bound
assert abs(P.prob("watch", 30, 0, "shallow") - P.prob("watch", 80, 0, "shallow")) < abs(P.prob("watch", 30, 0, "steep") - P.prob("watch", 80, 0, "steep"))
# categorical-age variant: constant within a band, device use falls and engagement rises across bands, age capped at 90
assert P.prob("watch", 66, 0, "band") == P.prob("watch", 74, 0, "band") and P.prob("watch", 10, 1, "band") == P.prob("watch", 34, 1, "band")
for dev in ("phone", "watch"):
    assert P.prob(dev, 70, 0, "band") > P.prob(dev, 80, 0, "band") > P.prob(dev, 88, 0, "band"), dev
assert P.prob("shared", 30, 0, "band") < P.prob("shared", 80, 0, "band")
assert P.prob("watch", 90, 0, "band") == P.prob("watch", 300, 0, "band")
pb = A.build_pool("repr_aware", age, fem, (5, 2, 0), dict(device="watch", engage="tilt", boost=1.0, repair_target="pool", age_bound="band"))
assert pb["n_recruited"] > 0

for mode in ("none", "tilt", "consent"):
    for dev in ("watch", "phone"):
        cfg = dict(device=dev, engage=mode, boost=1.0, repair_target="pool")
        pools = {a: A.build_pool(a, age, fem, (5, 2, 0), cfg) for a in A.ARMS}
        assert pools["uniform"]["eligible"].all()
        assert pools["size_matched"]["eligible"].sum() == pools["availability"]["eligible"].sum()
        assert (pools["post_hoc"]["eligible"] == pools["availability"]["eligible"]).all()          # common random numbers
        assert pools["repr_aware"]["eligible"][pools["availability"]["eligible"]].all()           # recruitment only adds
        assert (pools["availability"]["select_w"][~pools["availability"]["eligible"]] == 0).all()   # absent = never selected
        e = pools["post_hoc"]["eligible"]; w = pools["post_hoc"]["loss_w"][e]
        incl = P.prob(dev, age, fem) * (P.prob("shared", age, fem) if mode == "consent" else 1.0)   # P(patient is in the pool)
        assert abs(w.mean() - 1) < 1e-9
        assert np.corrcoef(w, incl[e])[0, 1] < -0.8        # weights are inverse to inclusion probability
        # (so for phone + consent, where engagement over-includes the elderly, they are down-weighted: correct)
        again = A.build_pool("availability", age, fem, (5, 2, 0), cfg)                              # reproducible
        assert (again["eligible"] == pools["availability"]["eligible"]).all()
        other = A.build_pool("availability", age, fem, (6, 2, 0), cfg)                              # seed-sensitive
        assert (other["eligible"] != pools["availability"]["eligible"]).any()

# repair dose-response: more boost, more recruits, and the sampled stream moves toward the population share
cfg = dict(device="watch", engage="none", repair_target="pool")
rec = [A.build_pool("repr_aware", age, fem, (1, 0, 0), dict(cfg, boost=b))["n_recruited"] for b in (0, 0.5, 1.0)]
assert rec[0] == 0 and rec[0] < rec[1] < rec[2], rec
pop65 = (age >= 65).mean()
s = [A.pool_summary(A.build_pool("repr_aware", age, fem, (1, 0, 0), dict(cfg, boost=b)), age)["share65"] for b in (0, 1.0)]
assert abs(s[1] - pop65) < abs(s[0] - pop65) and abs(s[1] - pop65) < 0.01, (s, pop65)   # pool target levels the POOL share
# stream target: with engage="tilt" the STREAM is levelled instead
cs = dict(device="watch", engage="tilt", repair_target="stream")
st = A.pool_summary(A.build_pool("repr_aware", age, fem, (1, 0, 0), dict(cs, boost=1.0)), age)["stream65"]
assert abs(st - pop65) < 0.01, (st, pop65)
# pool target and stream target recruit differently under tilt with the smartphone
cp = dict(device="phone", engage="tilt")
npool = A.build_pool("repr_aware", age, fem, (1, 0, 0), dict(cp, repair_target="pool", boost=1.0))["n_recruited"]
nstr = A.build_pool("repr_aware", age, fem, (1, 0, 0), dict(cp, repair_target="stream", boost=1.0))["n_recruited"]
assert npool != nstr
# repair_target has no default
try:
    A.build_pool("repr_aware", age, fem, (1, 0, 0), dict(device="watch", engage="none")); raise SystemExit("missing repair_target accepted")
except KeyError:
    pass
# availability pool under-represents the elderly when device use falls with age
assert A.pool_summary(A.build_pool("availability", age, fem, (1, 0, 0), cfg), age)["share65"] < pop65
print("test_participation.py: all invariants hold")
