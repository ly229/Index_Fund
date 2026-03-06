#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Ensure we can import the core IV helper.
sys.path.append("analysis/IV_result/IV_codes")
import panel_iv_rank_cutoff as piv

GRID_PATH = Path("analysis/IV_robutness_check/iv_robustness_grid_results.csv")
OUT_FIG = Path("analysis/IV_robutness_check/iv_local_windows_fitted.svg")


def select_specs():
    df = pd.read_csv(GRID_PATH)
    df = df[df["bandwidth"].notna()].copy()
    df["bandwidth"] = df["bandwidth"].astype(float)
    agg = (
        df.groupby(["cutoff", "bandwidth"], as_index=False)["first_stage_f"]
        .mean()
        .sort_values("first_stage_f")
        .reset_index(drop=True)
    )
    if len(agg) < 5:
        raise ValueError("Need at least five local specifications.")
    idxs = [0, 1, len(agg) // 2, len(agg) - 2, len(agg) - 1]
    specs = []
    for i in idxs:
        row = agg.loc[i]
        specs.append(
            {
                "cutoff": float(row["cutoff"]),
                "bandwidth": float(row["bandwidth"]),
                "avg_f": float(row["first_stage_f"]),
            }
        )
    return specs


def compute_fitted(dep_var: str, cutoff: float, bandwidth: float):
    y, x, z, w, firm_ids, time_ids = piv.prepare_dataset(
        dep_var, rank_map, cutoff, bandwidth
    )
    if len(y) == 0:
        raise RuntimeError(f"No observations for spec cutoff={cutoff}, bandwidth={bandwidth}")
    M = np.column_stack([y, x, z, w])
    M_dm = piv.two_way_demean(M, firm_ids, time_ids)
    y_dm = M_dm[:, 0]
    x_dm = M_dm[:, 1]
    z_dm = M_dm[:, 2]
    w_dm = M_dm[:, 3:]
    beta, _ = piv.iv_2sls_clustered(y_dm, x_dm, z_dm, w_dm, firm_ids)
    b = float(beta[0])

    xlo = float(np.quantile(x_dm, 0.01))
    xhi = float(np.quantile(x_dm, 0.99))
    if not np.isfinite(xlo) or not np.isfinite(xhi) or xhi <= xlo:
        xlo = float(np.nanmin(x_dm))
        xhi = float(np.nanmax(x_dm))
    if xhi <= xlo:
        xhi = xlo + 1.0

    return {
        "beta": b,
        "xlo": xlo,
        "xhi": xhi,
        "n": int(len(x_dm)),
    }


def main():
    global rank_map
    rank_map, _ = piv.load_rank_cutoff_map(piv.RANK_PATH)
    specs = select_specs()
    labels = [
        f"cutoff={int(s['cutoff'])}, bw={int(s['bandwidth'])}, F≈{s['avg_f']:.3f}"
        for s in specs
    ]

    fitted = {dep: [] for dep in piv.DEP_VARS}
    for spec in specs:
        for dep in piv.DEP_VARS:
            fitted[dep].append(
                compute_fitted(dep, spec["cutoff"], spec["bandwidth"])
            )

    fig, axes = plt.subplots(
        len(piv.DEP_VARS),
        1,
        figsize=(8.5, 3.7 * len(piv.DEP_VARS)),
        sharex=False,
    )
    if len(piv.DEP_VARS) == 1:
        axes = [axes]

    Y_LABELS = {
        "amihud_illiq": "amihud_illiq",
        "volatility": "volatility",
        "price_info": "price_info",
    }

    for ax, dep in zip(axes, piv.DEP_VARS):
        for spec_label, sound in zip(labels, fitted[dep]):
            x_line = np.linspace(sound["xlo"], sound["xhi"], 250)
            y_line = sound["beta"] * x_line
            ax.plot(x_line, y_line, label=spec_label, linewidth=2)
        ax.axhline(0.0, color="#888888", linestyle="--", linewidth=1)
        ax.set_ylabel(Y_LABELS.get(dep, dep))
        ax.grid(axis="y", alpha=0.25, linestyle=":")

    axes[-1].set_xlabel("demeaned ind_own")
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.25, right=0.82)
    legend = fig.legend(
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.08),
        fontsize="small",
        frameon=False,
        ncol=1,
    )
    note = (
        "Worst 2 (cutoff=2000, bandwidth=50/100) → median (cutoff=2000, bandwidth=300) → "
        "best 2 (cutoff=1000, bandwidth=50/100); lines show fitted slope (beta × demeaned ind_own)."
    )
    fig.text(0.5, -0.16, note, ha="center", va="top", fontsize="small")
    legend.set_zorder(10)
    fig.savefig(OUT_FIG, format="svg", dpi=220, bbox_inches="tight")
    print(f"Saved: {OUT_FIG}")


if __name__ == "__main__":
    main()
