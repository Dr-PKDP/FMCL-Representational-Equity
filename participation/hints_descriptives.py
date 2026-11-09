"""
hints_descriptives.py: weighted descriptives and an adjustment check for the HINTS 7 participation models.

    python3 hints_descriptives.py --sav <hints7_public.sav> --out <results/participation/hints_descriptives.json>

by_age_band: weighted percentage answering yes (outcome code 1 among codes 1 and 2) by age band, with the number of respondents.
adjusted: age coefficient of the smartphone and wearable models on the complete-case sample (valid IncomeRanges 1-9 and
Education 1-7), fitted with age and sex only and with household income range and education level added as linear terms.
Weights are PERSON_FINWT0, as in fit_hints.py.
"""
import argparse
import json

import numpy as np
import pandas as pd
import pyreadstat
import statsmodels.api as sm

OUTCOMES = {"phone": "UseDevice_SmPhone", "watch": "UseDevice_SmWatch", "shared": "SharedHealthDeviceInfo2"}
W = "PERSON_FINWT0"
BANDS = [("under 65", 0, 65), ("65 to 74", 65, 75), ("75 to 89", 75, 90), ("90 and over", 90, 200)]


def valid(df, y):
    return df[df[y].isin([1, 2]) & (df[W] > 0) & (df.Age > 0)]


def coef(d, y, extra):
    X = [np.minimum(d.Age.values, 90.0), (d.BirthSex.values == 1).astype(float)]
    if extra:
        X += [d.IncomeRanges.values, d.Education.values]
    fit = sm.GLM((d[y].values == 1).astype(float), sm.add_constant(np.column_stack(X)), family=sm.families.Binomial(),
                 var_weights=d[W].values).fit()
    return float(fit.params[1])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--sav", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cols = ["Age", "BirthSex", "IncomeRanges", "Education", W] + list(OUTCOMES.values())
    df, _ = pyreadstat.read_sav(a.sav, usecols=cols)
    res = {"by_age_band": {}, "adjusted": {}}
    for name, y in OUTCOMES.items():
        d = valid(df, y)
        res["by_age_band"][name] = {b: dict(pct=round(float(100 * (g[W] * (g[y] == 1)).sum() / g[W].sum()), 1), n=int(len(g)))
                                    for b, lo, hi in BANDS for g in [d[(d.Age >= lo) & (d.Age < hi)]]}
    for name in ("phone", "watch"):
        y = OUTCOMES[name]
        d = valid(df, y); d = d[d.BirthSex.isin([1, 2]) & d.IncomeRanges.between(1, 9) & d.Education.between(1, 7)]
        b0, b1 = coef(d, y, False), coef(d, y, True)
        res["adjusted"][name] = dict(n=int(len(d)), b_age_age_sex=round(b0, 4), b_age_with_income_education=round(b1, 4),
                                     change_pct=round(100 * (b1 / b0 - 1), 1))
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
