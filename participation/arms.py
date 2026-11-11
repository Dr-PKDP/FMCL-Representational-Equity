"""
arms.py

Participation arms. Eligibility is a PERSISTENT per-patient property drawn once per
(seed, fold, set): a patient who is not eligible contributes nothing in any round. This is
the eligibility/selection distinction: no weight can recover a patient
who was never in the pool. Engagement acts only as a per-round selection tilt among the
eligible.

Common random numbers: one uniform vector u (eligibility) and one v (recruitment) are drawn
per set and shared by arms 2, 3 and 4, so those arms differ only by the intervention.

    uniform        every patient eligible, equal selection weight        (reference)
    size_matched   random subset, same size as the availability pool     (separates composition
                   from the loss of data that eligibility causes)
    availability   eligible iff u < P(own device | age, sex) (engage="consent": u < P(own)*P(engage));
                   selection weight = P(engage) only when engage="tilt", otherwise uniform
    repr_aware     availability pool plus recruited patients, re-levelling the age strata to the population
                   shares (strength `boost` in [0,1]); recruits only add. The number recruited is the
                   carbon-relevant cost. cfg["repair_target"] (no default) chooses what is re-levelled:
                   "pool"   the composition of the enrolled pool (patient counts); an unrecruited patient is
                            the loss being repaired. Primary (protocol/analysis_plan.md).
                   "stream" the composition of the selection-weighted stream; with engage="tilt" and the
                            smartphone the stream already over-represents the elderly. Reported as a sensitivity.
    post_hoc       availability pool, unchanged; per-patient loss weights = 1 / P(own)
"""
import numpy as np
import propensity as P

ARMS = ["uniform", "size_matched", "availability", "repr_aware", "post_hoc"]
CONTROL_ARMS = ["recruit_random"]   # availability pool plus as many randomly chosen recruits as repr_aware recruits (volume control)
BANDS = [(0, 65), (65, 75), (75, 200)]          # age strata used for re-levelling


def _band(age):
    age = np.asarray(age, dtype=float)
    return np.digitize(age, [65, 75])           # 0: <65, 1: 65-74, 2: 75+


def build_pool(arm, age, female, seed_key, cfg):
    """
    age, female: arrays for the patients of one set (train or validation).
    seed_key: tuple of ints; identical keys give identical draws (reproducibility, and common
    random numbers across arms).
    cfg: dict with 'device' ('watch'|'phone'), 'engage' ('none'|'tilt'|'consent'), 'age_bound' (None|'shallow'|'steep'),
         'boost' (float, arm repr_aware), 'repair_target' ('pool'|'stream', arm repr_aware only).
    Returns dict: eligible (bool mask), select_w (relative selection weight, 0 outside pool),
                  loss_w (per-patient loss weight, 1 outside post_hoc), n_recruited.
    """
    age = np.asarray(age, dtype=float); female = np.asarray(female, dtype=float)
    n = len(age)
    rng_u = np.random.default_rng(list(seed_key) + [1])
    rng_v = np.random.default_rng(list(seed_key) + [2])
    rng_s = np.random.default_rng(list(seed_key) + [3])
    u = rng_u.random(n); v = rng_v.random(n)

    # cfg["engage"]: how the engagement model (willingness to share health-device data) enters
    #   "none"    eligibility = device use only; uniform selection among the eligible
    #   "tilt"    device-use eligibility; engagement tilts per-round selection among the eligible
    #   "consent" engagement is a persistent consent condition: eligible iff u < P(own) * P(engage)
    mode = cfg["engage"]          # no default on purpose: the engagement reading is a pre-specification decision
    own = P.prob(cfg["device"], age, female, cfg.get("age_bound"))
    eng_p = P.prob("shared", age, female, cfg.get("age_bound"))
    eng = eng_p if mode == "tilt" else np.ones(n)
    elig_av = u < (own * eng_p if mode == "consent" else own)
    out = dict(n_recruited=0, loss_w=np.ones(n))

    if arm == "uniform":
        elig = np.ones(n, dtype=bool); sel = np.ones(n)
    elif arm == "size_matched":
        k = int(elig_av.sum())
        elig = np.zeros(n, dtype=bool); elig[rng_s.choice(n, size=k, replace=False)] = True
        sel = np.ones(n)
    elif arm in ("availability", "post_hoc"):
        elig = elig_av; sel = eng.copy()
        if arm == "post_hoc":
            incl = own * eng_p if mode == "consent" else own     # probability of being in the pool
            w = 1.0 / np.clip(incl, 1e-3, None)
            out["loss_w"] = w / w[elig].mean()
    elif arm == "repr_aware":
        # Re-level age-stratum composition. The "mass" of a stratum is the number of eligible patients
        # (repair_target="pool") or the sum of their selection weights (repair_target="stream").
        # Final total mass M is set by the most over-represented stratum (mass only grows); stratum k
        # then needs share_k * M - mass_k more mass, scaled by boost (0 = none, 1 = full), filled by
        # recruiting patients in the order of the common random vector v. Recruits count n_recruited,
        # which is the carbon-relevant cost.
        target = cfg["repair_target"]             # no default on purpose (pre-specified in the analysis plan)
        if target not in ("pool", "stream"):
            raise ValueError(target)
        mw = np.ones(n) if target == "pool" else eng
        elig = elig_av.copy(); band = _band(age)
        share = np.bincount(band, minlength=3) / n
        mass = np.array([mw[elig & (band == k)].sum() for k in range(3)])
        M_final = max(mass[k] / share[k] for k in range(3) if share[k] > 0)
        boost = cfg.get("boost", 1.0)
        for k in range(3):
            need = boost * max(0.0, share[k] * M_final - mass[k])
            cand = np.where((band == k) & ~elig)[0]
            if need > 0 and len(cand):
                order = cand[np.argsort(v[cand])]
                cum = np.cumsum(mw[order])
                m = int(np.searchsorted(cum, need)) + 1          # fewest patients reaching the needed mass
                pick = order[:min(m, len(order))]
                elig[pick] = True; out["n_recruited"] += len(pick)
        sel = eng.copy()
    elif arm == "recruit_random":
        # Volume control for repr_aware: the same number of recruits, drawn without regard to age or sex.
        k = build_pool("repr_aware", age, female, seed_key, cfg)["n_recruited"]
        elig = elig_av.copy()
        cand = np.where(~elig)[0]
        pick = cand[np.argsort(v[cand])][:k]
        elig[pick] = True; out["n_recruited"] = len(pick)
        sel = eng.copy()
    else:
        raise ValueError(arm)

    out["eligible"] = elig
    out["select_w"] = np.where(elig, sel, 0.0)
    return out


def pool_summary(pool, age):
    e = pool["eligible"]; b = _band(age)
    n = max(int(e.sum()), 1)
    w = pool["select_w"]; W = max(float(w.sum()), 1e-12)
    return dict(n_eligible=int(e.sum()), n_recruited=int(pool["n_recruited"]),
                share65=float(((b >= 1) & e).sum() / n), share75=float(((b == 2) & e).sum() / n),
                stream65=float(w[b >= 1].sum() / W), stream75=float(w[b == 2].sum() / W))
