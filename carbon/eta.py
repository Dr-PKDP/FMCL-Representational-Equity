"""
eta.py: carbon-equity frontier (Eq. 5) from recruitment quantities and the paired sensitivity contrasts.

    python3 eta.py --analysis-dir <dir with recruit_stats.csv, frontier_contrasts.csv, volume_control_contrasts.csv>
                   [--kappa 1.5 2 3 5 10] [--subgroup age65] [--suite val_common | --suite sens_band --recruit-file recruit_stats_band.csv]

Model. Training volume is matched across arms, so with a participation carbon c_p (kg CO2e for one participation by an
ordinary device) the campaign carbon of an arm exceeds the availability arm's only through the participations of its
recruited devices, which cost kappa x c_p each (kappa >= 1: onboarding, outreach and device-specific overhead of a recruited participation):
        Phi(a) - Phi(avail) = (kappa - 1) x R_rec(a) x c_p  +  n_rec(a) x eps x C_emb
with R_rec the expected participations by recruits, n_rec the recruits, eps the share of a recruit device's embodied
carbon C_emb attributed to the campaign (Eq. 3; eps = 0 under assumption A2). Eq. 5:
        eta(a) = [Phi(a) - Phi(avail)] / [Sens(a) - Sens(avail)]    per percentage point.
The table reports eta in units of c_p (participation-equivalents per percentage point) for each kappa, so the physical
value follows by multiplying with c_p; the embodied term is reported separately as recruits per percentage point, to be
multiplied by eps x C_emb. The interval of eta comes from the paired-bootstrap interval of the sensitivity contrast; where that
interval includes zero, eta has no finite upper limit and is reported as unbounded.
"""
import argparse, os
import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--analysis-dir", required=True); ap.add_argument("--subgroup", default="age65")
    ap.add_argument("--kappa", type=float, nargs="+", default=[1.5, 2, 3, 5, 10])
    ap.add_argument("--suite", default=None, help="take the contrasts from <suite>_contrasts.csv (e.g. val_common, sens_band) "
                    "instead of the frontier and volume-control suites; output eta_<subgroup>_<suite>.csv")
    ap.add_argument("--recruit-file", default="recruit_stats.csv")
    a = ap.parse_args()
    rs = pd.read_csv(os.path.join(a.analysis_dir, a.recruit_file)).set_index("label")
    if a.suite:
        sc = pd.read_csv(os.path.join(a.analysis_dir, f"{a.suite}_contrasts.csv"))
        con = sc[(sc.b == "availability") & (sc.a.str.startswith("repr_aware") | (sc.a == "recruit_random"))]
    else:
        fr = pd.read_csv(os.path.join(a.analysis_dir, "frontier_contrasts.csv"))
        vc = pd.read_csv(os.path.join(a.analysis_dir, "volume_control_contrasts.csv"))
        con = pd.concat([fr[fr.b == "availability"], vc[(vc.a == "recruit_random") & (vc.b == "availability")]], ignore_index=True)
    con = con[con.subgroup == a.subgroup].drop_duplicates(["a", "b", "subgroup"])
    rows = []
    for _, c in con.iterrows():
        if c.a not in rs.index: continue
        r = rs.loc[c.a]; d, lo, hi = 100 * c["diff"], 100 * c.lo95, 100 * c.hi95        # percentage points
        row = dict(arm=c.a, n_rec=r.n_rec, R_rec=r.R_rec, rec_age65=r.rec_age65, d_pp=d, d_lo=lo, d_hi=hi,
                   recruits_per_pp=r.n_rec / d if d > 0 else np.nan,
                   recruits_per_pp_lo=r.n_rec / hi if lo > 0 else np.nan, recruits_per_pp_hi=r.n_rec / lo if lo > 0 else np.inf)
        for k in a.kappa:
            row[f"eta_k{k:g}"] = (k - 1) * r.R_rec / d if d > 0 else np.nan
            row[f"eta_k{k:g}_lo"] = (k - 1) * r.R_rec / hi if lo > 0 else np.nan
            row[f"eta_k{k:g}_hi"] = (k - 1) * r.R_rec / lo if lo > 0 else np.inf
        rows.append(row)
    out = pd.DataFrame(rows)
    order = ["repr_aware@0.25", "repr_aware@0.5", "repr_aware@0.75", "repr_aware", "recruit_random"]
    out["o"] = out.arm.map({k: i for i, k in enumerate(order)}); out = out.sort_values("o").drop(columns="o")
    out.to_csv(os.path.join(a.analysis_dir, f"eta_{a.subgroup}" + (f"_{a.suite}" if a.suite else "") + ".csv"), index=False)
    pd.set_option("display.width", 220, "display.max_columns", 40)
    print(out.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
