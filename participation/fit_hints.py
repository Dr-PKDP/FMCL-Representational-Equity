"""
fit_hints.py: participation models from the HINTS 7 (2024) public-use file.

    python3 fit_hints.py --sav <hints7_public.sav> [--out models.json]

Weighted logistic regressions (statsmodels GLM, variance weights = PERSON_FINWT0) of
    phone    UseDevice_SmPhone          use of a smartphone
    watch    UseDevice_SmWatch          use of a smartwatch or wearable
    shared   SharedHealthDeviceInfo2    shared data from a health device with a clinician in the past year
on age (capped at 90) and sex (BirthSex: 1 = female, 2 = male). Outcome codes 1 = yes and 2 = no; other codes and negative
survey missing codes are dropped. Standard errors: delete-one jackknife over the 50 replicate weights PERSON_FINWT1..50,
v = (R - 1) / R * sum_r (b_r - b)^2 with R = 50. The output is the content of MODELS in propensity.py.
A second fit per outcome replaces linear age by six age bands (18-34, 35-49, 50-64, 65-74, 75-84, 85-90; one log-odds per
band, no intercept) with the same sex term: the content of BAND_MODELS in propensity.py (functional-form sensitivity).
"""
import argparse
import json

import numpy as np
import pandas as pd
import pyreadstat
import statsmodels.api as sm

OUTCOMES = {"phone": "UseDevice_SmPhone", "watch": "UseDevice_SmWatch", "shared": "SharedHealthDeviceInfo2"}
R = 50


def fit(df, y, w):
    d = df[(df.Age > 0) & df.BirthSex.isin([1, 2]) & df[y].isin([1, 2]) & (df[w] > 0)]
    X = sm.add_constant(np.column_stack([np.minimum(d.Age.values, 90.0), (d.BirthSex.values == 1).astype(float)]))
    return np.asarray(sm.GLM((d[y].values == 1).astype(float), X, family=sm.families.Binomial(), var_weights=d[w].values).fit().params)


BAND_EDGES = [35, 50, 65, 75, 85]


def fit_band(df, y, w):
    d = df[(df.Age > 0) & df.BirthSex.isin([1, 2]) & df[y].isin([1, 2]) & (df[w] > 0)]
    b = np.digitize(np.minimum(d.Age.values, 90.0), BAND_EDGES)
    X = np.column_stack([np.eye(len(BAND_EDGES) + 1)[b], (d.BirthSex.values == 1).astype(float)])
    return np.asarray(sm.GLM((d[y].values == 1).astype(float), X, family=sm.families.Binomial(), var_weights=d[w].values).fit().params)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--sav", required=True); ap.add_argument("--out", default=None)
    a = ap.parse_args()
    cols = ["Age", "BirthSex"] + list(OUTCOMES.values()) + [f"PERSON_FINWT{i}" for i in range(R + 1)]
    df, _ = pyreadstat.read_sav(a.sav, usecols=cols)
    res = {}
    for name, y in OUTCOMES.items():
        b = fit(df, y, "PERSON_FINWT0")
        reps = np.array([fit(df, y, f"PERSON_FINWT{r}") for r in range(1, R + 1)])
        se = np.sqrt((R - 1) / R * ((reps - b) ** 2).sum(axis=0))
        res[name] = dict(intercept=b[0], age=b[1], female=b[2], se_age=se[1], se_female=se[2],
                         ci_age=(b[1] - 1.96 * se[1], b[1] + 1.96 * se[1]))
        print(f"{name:7s} b0 {b[0]:+.4f}  age {b[1]:+.5f} (SE {se[1]:.5f}; CI {res[name]['ci_age'][0]:+.4f}, {res[name]['ci_age'][1]:+.4f})  female {b[2]:+.4f} (SE {se[2]:.4f})")
        bb = fit_band(df, y, "PERSON_FINWT0"); res[name]["band"] = dict(logodds=list(bb[:-1]), female=bb[-1])
        print(f"{name:7s} band log-odds " + " ".join(f"{v:+.4f}" for v in bb[:-1]) + f"  female {bb[-1]:+.4f}")
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
