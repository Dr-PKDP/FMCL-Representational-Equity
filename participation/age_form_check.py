"""
age_form_check.py: functional form of age in the HINTS 7 participation models.

    python3 age_form_check.py --sav <hints7_public.sav> --out <csv>

For each outcome (phone, watch, shared) the weighted observed proportion in six age bands is compared with the fitted
probability of the model that is linear in age on the logit scale (the model used in the study) and of a model with
categorical age bands (same bands, same sex term), both fitted with the final survey weight and averaged over the
weighted band population, and with the fitted probability of a model with a natural cubic spline of age (4 degrees of
freedom). The column steep is the probability of the steep-coefficient sensitivity model of propensity.py (age
coefficient at the confidence limit furthest from zero) at the respondents' ages and sexes. Band proportions of the three models are on the probability scale. Standard errors of the observed
proportions use the 50 replicate weights (delete-one jackknife, factor (R-1)/R).
"""
import argparse
import numpy as np
import pandas as pd
import os
import sys
import pyreadstat
import statsmodels.api as sm
from patsy import dmatrix

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import propensity as P

OUT = {"phone": "UseDevice_SmPhone", "watch": "UseDevice_SmWatch", "shared": "SharedHealthDeviceInfo2"}
BANDS = [(18, 35), (35, 50), (50, 65), (65, 75), (75, 85), (85, 91)]
R = 50


def design(kind, age, fem):
    if kind == "linear":
        return sm.add_constant(np.column_stack([age, fem]))
    if kind == "band":
        b = np.digitize(age, [lo for lo, _ in BANDS[1:]])
        return np.column_stack([np.eye(len(BANDS))[b], fem])
    return np.asarray(dmatrix("cr(age, df=4) - 1 + fem", {"age": age, "fem": fem}, return_type="dataframe"))


def fitted_bands(d, y, w, kind):
    age = np.minimum(d.Age.values, 90.0); fem = (d.BirthSex.values == 1).astype(float)
    X = design(kind, age, fem)
    m = sm.GLM((d[y].values == 1).astype(float), X, family=sm.families.Binomial(), var_weights=d[w].values).fit()
    p = m.predict(X); out = []
    for lo, hi in BANDS:
        s = (age >= lo) & (age < hi); out.append(np.average(p[s], weights=d[w].values[s]))
    return np.array(out)


def steep_bands(d, name, w):
    age = np.minimum(d.Age.values, 90.0); fem = (d.BirthSex.values == 1).astype(float)
    p = P.prob(name, age, fem, "steep")
    return np.array([np.average(p[(age >= lo) & (age < hi)], weights=d[w].values[(age >= lo) & (age < hi)]) for lo, hi in BANDS])


def observed(d, y, w):
    age = np.minimum(d.Age.values, 90.0); yy = (d[y].values == 1).astype(float)
    return np.array([np.average(yy[(age >= lo) & (age < hi)], weights=d[w].values[(age >= lo) & (age < hi)]) for lo, hi in BANDS])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--sav", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cols = ["Age", "BirthSex"] + list(OUT.values()) + [f"PERSON_FINWT{i}" for i in range(R + 1)]
    df, _ = pyreadstat.read_sav(a.sav, usecols=cols)
    rows = []
    for name, y in OUT.items():
        d = df[(df.Age > 0) & df.BirthSex.isin([1, 2]) & df[y].isin([1, 2]) & (df.PERSON_FINWT0 > 0)]
        obs = observed(d, y, "PERSON_FINWT0")
        reps = np.array([observed(d[d.PERSON_FINWT0 > 0], y, f"PERSON_FINWT{r}") for r in range(1, R + 1)])
        se = np.sqrt((R - 1) / R * ((reps - obs) ** 2).sum(axis=0))
        st = steep_bands(d, name, "PERSON_FINWT0")
        fits = {k: fitted_bands(d, y, "PERSON_FINWT0", k) for k in ("linear", "band", "spline")}
        for i, (lo, hi) in enumerate(BANDS):
            rows.append(dict(outcome=name, age_lo=lo, age_hi=hi - 1 if hi < 91 else 90, n=int(((np.minimum(d.Age.values, 90.0) >= lo) & (np.minimum(d.Age.values, 90.0) < hi)).sum()),
                             observed=obs[i], se=se[i], linear=fits["linear"][i], band=fits["band"][i], spline=fits["spline"][i], steep=st[i]))
    t = pd.DataFrame(rows); t.to_csv(a.out, index=False)
    pd.set_option("display.width", 200); print(t.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
