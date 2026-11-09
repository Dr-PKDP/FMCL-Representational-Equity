"""Figures drawn from ledger.json (build_ledger.py). Writes SVG and PNG into fig/."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

P = Path(__file__).resolve().parent
R = P.parent / "results"
F = P / "fig"
F.mkdir(exist_ok=True)
L = json.load(open(P / "ledger.json"))
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "text.color": "black",
                     "axes.labelcolor": "black", "xtick.color": "black", "ytick.color": "black",
                     "axes.edgecolor": "black", "svg.fonttype": "none", "axes.linewidth": 0.6})
ROWS = [("Pool size", "size_matched", "uniform"),
        ("Composition", "availability", "size_matched"),
        ("Eligibility total", "availability", "uniform"),
        ("Added volume", "recruit_random", "availability"),
        ("Targeting", "repr_aware", "recruit_random"),
        ("Repair total", "repr_aware", "availability")]
PANELS = [("a", "A.primary", "A.volume_control", "age65", "PTB-XL, age 65+"),
          ("b", "A.ext_code15", None, "age65", "CODE-15%, age 65+"),
          ("c", "A.ext_chapman", None, "age65", "Chapman and Ningbo, age 65+"),
          ("d", "B.primary", "B.volume", "age65", "PAD-UFES-20, age 65+")]


def get(pre, vpre, a, b, sg):
    for p in (pre, vpre):
        if p and f"{p}.{a}_vs_{b}.{sg}" in L:
            return L[f"{p}.{a}_vs_{b}.{sg}"]
    return None


def fig1():
    fig, axs = plt.subplots(1, 4, figsize=(7.4, 2.7), sharey=True)
    for ax, (tag, pre, vpre, sg, title) in zip(axs, PANELS):
        for i, (lab, a, b) in enumerate(ROWS):
            e = get(pre, vpre, a, b, sg)
            y = len(ROWS) - 1 - i
            if e is None:
                continue
            ax.errorbar(100 * e["diff"], y, xerr=[[100 * (e["diff"] - e["lo"])], [100 * (e["hi"] - e["diff"])]],
                        fmt="s", ms=3.2, color="black", capsize=2, lw=0.8)
        ax.axvline(0, color="black", lw=0.5, ls=":")
        ax.set_title(title, fontsize=7.5)
        ax.set_yticks(range(len(ROWS)))
        ax.set_yticklabels([r[0] for r in ROWS][::-1])
        ax.set_xlabel("Difference (points)")
        ax.tick_params(length=2)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    plt.tight_layout()
    for ext in ("svg", "png"):
        plt.savefig(F / f"fig1.{ext}", dpi=300)
    plt.close()


def fig2():
    labs = ["repr_aware@0.25", "repr_aware@0.5", "repr_aware@0.75", "repr_aware", "recruit_random"]
    fig, axs = plt.subplots(1, 2, figsize=(7.4, 3.0))
    ax = axs[0]
    xs = [L[f"A.recruit.{l}.n_rec"] for l in labs]
    for l, x in zip(labs, xs):
        d = L[f"A.eta.{l}.d_pp"]
        ax.errorbar(x, d, yerr=[[d - L[f"A.eta.{l}.d_lo"]], [L[f"A.eta.{l}.d_hi"] - d]],
                    fmt="o" if l != "recruit_random" else "^", ms=4, color="black",
                    mfc="black" if l != "recruit_random" else "white", capsize=2, lw=0.8)
    fx = [0] + xs[:4]
    fy = [0] + [L[f"A.eta.{l}.d_pp"] for l in labs[:4]]
    ax.plot(fx, fy, color="black", lw=0.7)
    ax.set_xlabel("Recruited patients per training pool")
    ax.set_ylabel("Recovered 65+ sensitivity (points)")
    hA = [plt.Line2D([], [], marker="o", color="black", ms=4, lw=0.7, label="Age-targeted"),
          plt.Line2D([], [], marker="^", mfc="white", color="black", ms=4, lw=0, label="Random")]
    ax.set_title("A", loc="left", fontsize=8, fontweight="bold")
    ax = axs[1]
    ks = [1.5, 2, 3, 5, 10]
    for l, mk, ls in (("repr_aware@0.75", "o", "-"), ("repr_aware", "s", "--"), ("recruit_random", "^", ":")):
        ys = [L[f"A.eta.{l}.eta_k{k}"] / 1000 for k in ks]
        ax.plot(ks, ys, marker=mk, ms=3.5, color="black", lw=0.8, ls=ls, mfc="black" if l != "recruit_random" else "white",
                label={"repr_aware@0.75": "Targeted, boost 0.75", "repr_aware": "Targeted, boost 1",
                       "recruit_random": "Random"}[l])
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(ks)
    ax.set_xticklabels([str(k) for k in ks])
    ax.set_xlabel("Cost multiplier κ of a recruited participation")
    ax.set_ylabel("η (10³ per point)")
    hB = ax.get_legend_handles_labels()[0]
    from matplotlib.ticker import FixedLocator, FixedFormatter, NullFormatter, NullLocator
    ax.xaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_major_locator(FixedLocator([10, 30, 100, 300]))
    ax.yaxis.set_major_formatter(FixedFormatter(["10", "30", "100", "300"]))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.set_title("B", loc="left", fontsize=8, fontweight="bold")
    for a in axs:
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
    plt.tight_layout(rect=(0, 0.1, 1, 1))
    fig.legend(handles=hA, loc="lower left", bbox_to_anchor=(0.08, 0.0), frameon=False, ncol=2)
    fig.legend(handles=hB, loc="lower right", bbox_to_anchor=(0.98, 0.0), frameon=False, ncol=3, columnspacing=0.8, handlelength=1.6)
    for ext in ("svg", "png"):
        plt.savefig(F / f"fig2.{ext}", dpi=300)
    plt.close()


def fig3():
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.6))
    for ax, f, title, ls in ((axs[0], "dca_ecg_contrasts.csv", "A  PTB-XL, age 65+", None),
                             (axs[1], "dca_derm_contrasts.csv", "B  PAD-UFES-20, age 65+", None)):
        d = pd.read_csv(R / "dca" / f)
        for (a, b), lab, st in ((("availability", "uniform"), "Availability − uniform", "-"),
                                (("repr_aware", "availability"), "Repair − availability", "--"),
                                (("availability", "size_matched"), "Availability − size-matched", ":")):
            s = d[(d.a == a) & (d.b == b)]
            ax.plot(s.threshold, 100 * s["diff"], color="black", ls=st, lw=0.9, label=lab)
            ax.fill_between(s.threshold, 100 * s.lo95, 100 * s.hi95, color="black", alpha=0.12, lw=0)
        ax.axhline(0, color="black", lw=0.5)
        ax.set_xlabel("Threshold probability")
        ax.set_ylabel("Net benefit difference (×10⁻²)")
        ax.set_title(title, loc="left", fontsize=8, fontweight="bold")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", frameon=False, ncol=3)
    plt.tight_layout(rect=(0, 0.08, 1, 1))
    for ext in ("svg", "png"):
        plt.savefig(F / f"fig3.{ext}", dpi=300)
    plt.close()


fig1(); fig2(); fig3()
print("done")
