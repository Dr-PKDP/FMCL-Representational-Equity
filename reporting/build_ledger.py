"""Collect every reported number from the committed result tables.

Writes ledger.json (flat key -> value) and ledger.md (index) next to this script.
Key scheme: <task>.<suite>.<a>_vs_<b>.<subgroup> -> {diff, lo, hi, p, p_bh}
            <task>.<suite>.arm.<arm>.<field>
"""
import json
import math
from pathlib import Path

import pandas as pd

R = Path(__file__).resolve().parent.parent / "results"
OUT = Path(__file__).resolve().parent
L = {}


def put(k, v):
    if isinstance(v, float) and math.isnan(v):
        return
    L[k] = v


for task, sub, pre in (("A", "ecg_analysis", ""), ("B", "derm_analysis", "derm_")):
    for f in sorted((R / sub).glob("*_contrasts.csv")):
        suite = f.stem[: -len("_contrasts")]
        suite = suite[len(pre):] if pre and suite.startswith(pre) else suite
        for r in pd.read_csv(f).itertuples():
            k = f"{task}.{suite}.{r.a}_vs_{r.b}.{r.subgroup}"
            put(k, dict(diff=r.diff, lo=r.lo95, hi=r.hi95, lo90=getattr(r, "lo90", float("nan")), hi90=getattr(r, "hi90", float("nan")), p=getattr(r, "p_boot", float("nan")),
                        p_bh=getattr(r, "p_boot_bh", float("nan")), p_sf=getattr(r, "p_signflip", float("nan")), n_seeds=int(r.n_seeds)))
    for f in sorted((R / sub).glob("*_arms.csv")):
        suite = f.stem[: -len("_arms")]
        suite = suite[len(pre):] if pre and suite.startswith(pre) else suite
        for r in pd.read_csv(f).to_dict("records"):
            for c, v in r.items():
                if c != "arm":
                    put(f"{task}.{suite}.arm.{r['arm']}.{c}", v)
    for f in sorted((R / sub / "agegap").glob("*_contrasts.csv")):
        suite = f.stem[: -len("_contrasts")]
        suite = suite[len(pre):] if pre and suite.startswith(pre) else suite
        for r in pd.read_csv(f).itertuples():
            put(f"{task}.agegap.{suite}.{r.a}_vs_{r.b}.{r.quantity}",
                dict(diff=r.diff, lo=r.lo95, hi=r.hi95, lo90=r.lo90, hi90=r.hi90, p=r.p_boot, p_sf=r.p_signflip, n_seeds=int(r.n_seeds)))
    for f in sorted((R / sub / "agegap").glob("*_arms.csv")):
        suite = f.stem[: -len("_arms")]
        suite = suite[len(pre):] if pre and suite.startswith(pre) else suite
        for r in pd.read_csv(f).to_dict("records"):
            for c, v in r.items():
                if c != "arm":
                    put(f"{task}.agegap.{suite}.arm.{r['arm']}.{c}", v)
    for f in sorted((R / sub).glob("*_summary.json")):
        suite = f.stem[: -len("_summary")]
        suite = suite[len(pre):] if pre and suite.startswith(pre) else suite
        for c, v in json.load(open(f)).items():
            if not isinstance(v, list):
                put(f"{task}.{suite}.summary.{c}", v)

for r in pd.read_csv(R / "ecg_analysis" / "recruit_stats.csv").to_dict("records"):
    for c, v in r.items():
        if c != "label":
            put(f"A.recruit.{r['label']}.{c}", v)
for r in pd.read_csv(R / "ecg_analysis" / "eta_age65.csv").to_dict("records"):
    for c, v in r.items():
        if c != "arm":
            put(f"A.eta.{r['arm']}.{c}", v)
for f in sorted((R / "ecg_analysis").glob("recruit_stats_*.csv")):           # participation-model variants, e.g. recruit_stats_band
    if f.stem.endswith("_jobs"): continue
    for r in pd.read_csv(f).to_dict("records"):
        for c, v in r.items():
            if c != "label":
                put(f"A.recruit_{f.stem[len('recruit_stats_'):]}.{r['label']}.{c}", v)
for f in sorted((R / "ecg_analysis").glob("eta_age65_*.csv")):               # eta from the contrasts of one suite
    for r in pd.read_csv(f).to_dict("records"):
        for c, v in r.items():
            if c != "arm":
                put(f"A.eta_{f.stem[len('eta_age65_'):]}.{r['arm']}.{c}", v)
for task, sub, pre in (("A", "ecg_analysis", ""), ("B", "derm_analysis", "derm_")):  # seed-and-patient bootstrap
    for f in sorted((R / sub / "seedpatient").glob("*_contrasts.csv")):
        suite = f.stem[: -len("_contrasts")]; suite = suite[len(pre):] if pre and suite.startswith(pre) else suite
        for r in pd.read_csv(f).itertuples():
            put(f"{task}.sp.{suite}.{r.a}_vs_{r.b}.{r.quantity}", dict(diff=r.diff, lo=r.lo95_joint, hi=r.hi95_joint,
                                                                     lo_patient=r.lo95_patient, hi_patient=r.hi95_patient, p=r.p_joint, n_seeds=int(r.n_seeds)))
    for f in sorted((R / sub / "seedpatient").glob("*_eta.csv")):
        suite = f.stem[: -len("_eta")]; suite = suite[len(pre):] if pre and suite.startswith(pre) else suite
        for r in pd.read_csv(f).to_dict("records"):
            for c, v in r.items():
                if c != "arm":
                    put(f"{task}.sp_eta.{suite}.{r['arm']}.{c}", v)
for task, sub, pre in (("A", "ecg_analysis", ""), ("B", "derm_analysis", "derm_")):
    base = R / sub / "commonthr"                                               # one operating threshold per model
    for f in sorted(base.glob("*_contrasts.csv")) if base.exists() else []:
        suite = f.stem[: -len("_contrasts")]; suite = suite[len(pre):] if pre and suite.startswith(pre) else suite
        for r in pd.read_csv(f).itertuples():
            put(f"{task}.ct.{suite}.{r.a}_vs_{r.b}.{r.quantity}", dict(diff=r.diff, lo=r.lo95, hi=r.hi95, p=r.p_boot, n_seeds=int(r.n_seeds)))
    for f in sorted(base.glob("*_arms.csv")) if base.exists() else []:
        suite = f.stem[: -len("_arms")]; suite = suite[len(pre):] if pre and suite.startswith(pre) else suite
        for r in pd.read_csv(f).to_dict("records"):
            for c, v in r.items():
                if c != "arm":
                    put(f"{task}.ct_arm.{suite}.{r['arm']}.{c}", v)
    base = R / sub / "indep"                                                   # evaluation restricted to test folds 7-10
    for f in sorted(base.glob("*_contrasts.csv")) if base.exists() else []:
        suite = f.stem[: -len("_contrasts")]; suite = suite[len(pre):] if pre and suite.startswith(pre) else suite
        for r in pd.read_csv(f).itertuples():
            put(f"{task}.indep.{suite}.{r.a}_vs_{r.b}.{r.subgroup}", dict(diff=r.diff, lo=r.lo95, hi=r.hi95, p=getattr(r, "p_boot", float("nan")), n_seeds=int(r.n_seeds)))
    for f in sorted((base / "agegap").glob("*_contrasts.csv")) if base.exists() and (base / "agegap").exists() else []:
        suite = f.stem[: -len("_contrasts")]; suite = suite[len(pre):] if pre and suite.startswith(pre) else suite
        for r in pd.read_csv(f).itertuples():
            put(f"{task}.indep_gap.{suite}.{r.a}_vs_{r.b}.{r.quantity}", dict(diff=r.diff, lo=r.lo95, hi=r.hi95, p=r.p_boot, n_seeds=int(r.n_seeds)))
    for f in sorted(base.glob("*_arms.csv")) if base.exists() else []:
        suite = f.stem[: -len("_arms")]; suite = suite[len(pre):] if pre and suite.startswith(pre) else suite
        for r in pd.read_csv(f).to_dict("records"):
            for c, v in r.items():
                if c != "arm":
                    put(f"{task}.indep_arm.{suite}.{r['arm']}.{c}", v)
for task, nm in (("A", "ecg"), ("B", "derm")):                                  # hyperparameter selection restricted to folds 1-6
    f = R / "tuning" / f"selection_{nm}.json"
    if f.exists():
        d = json.load(open(f))
        for c in ("selected_val_loss", "val_loss_at_main_values", "grid_range", "n_cells", "same_as_main"):
            put(f"{task}.tune.{c}", d[c])
        for c, v in d["selected"].items(): put(f"{task}.tune.selected.{c}", v)
        for c, v in d["main_experiments"].items(): put(f"{task}.tune.main.{c}", v)
    g = R / "tuning" / f"tuning_{nm}_independent.csv"
    if g.exists():
        for r in pd.read_csv(g).to_dict("records"):
            put(f"{task}.tune.grid.lr{r['lr']:g}.ee{int(r['epoch_equiv'])}", r["mean_best_val_loss"])
for r in pd.read_csv(R / "ecg_analysis" / "primary_diagnostics.csv").to_dict("records"):
    for c, v in r.items():
        if c != "label":
            put(f"A.pool.{r['label']}.{c}", v)

for r in pd.read_csv(R / "derm_analysis" / "fitz_contrasts.csv").itertuples():
    put(f"B.fitz.{r.a}_vs_{r.b}.{r.subgroup}", dict(diff=r.diff, lo=r.lo95, hi=r.hi95))
for r in pd.read_csv(R / "derm_analysis" / "fitz_arms.csv").to_dict("records"):
    for c, v in r.items():
        if c != "arm":
            put(f"B.fitz.arm.{r['arm']}.{c}", v)

for task, f in (("A", "dca_ecg_contrasts.csv"), ("B", "dca_derm_contrasts.csv")):
    for r in pd.read_csv(R / "dca" / f).itertuples():
        put(f"{task}.dca.{r.a}_vs_{r.b}.t{r.threshold:.2f}", dict(diff=r.diff, lo=r.lo95, hi=r.hi95))

for r in pd.read_csv(R / "derm_analysis" / "derm_primary_diagnostics.csv").to_dict("records"):
    for c, v in r.items():
        if c != "label":
            put(f"B.pool.{r['label']}.{c}", v)

for r in pd.read_csv(R / "varcomp" / "varcomp_components.csv").to_dict("records"):
    for c, v in r.items():
        if c != "quantity":
            put(f"A.varcomp.{r['quantity']}.{c}", v)

json.dump(L, open(OUT / "ledger.json", "w"), indent=1, default=float)
with open(OUT / "ledger.md", "w") as fh:
    for k in sorted(L):
        v = L[k]
        if isinstance(v, dict):
            v = "{:+.4f} [{:+.4f}, {:+.4f}]".format(v["diff"], v["lo"], v["hi"])
        fh.write(f"{k}\t{v}\n")
print(len(L), "entries")
