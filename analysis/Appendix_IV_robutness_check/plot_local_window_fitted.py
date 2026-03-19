#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent.parent
IV_CODES_DIR = ROOT_DIR / "analysis" / "IV_result" / "IV_codes"
if str(IV_CODES_DIR) not in sys.path:
    sys.path.append(str(IV_CODES_DIR))

import panel_iv_rank_cutoff as piv


GRID_PATH = SCRIPT_DIR / "iv_robustness_grid_results.csv"
OUT_FIG = SCRIPT_DIR / "iv_local_windows_fitted.svg"

DEP_LABELS = {
    "amihud_illiq": "Amihud illiquidity",
    "volatility": "Volatility",
    "price_info": "Price informativeness",
}
PANEL_MARKS = {
    "amihud_illiq": "a)",
    "volatility": "b)",
    "price_info": "c)",
}


def set_research_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["STIX Two Text", "Times New Roman", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#333333",
            "axes.linewidth": 0.8,
            "axes.titlesize": 12.5,
            "axes.titleweight": "semibold",
            "axes.labelsize": 11,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "legend.fontsize": 9.2,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


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
        raise RuntimeError(
            f"No observations for spec cutoff={cutoff}, bandwidth={bandwidth}"
        )

    M = np.column_stack([y, x, z, w])
    M_dm = piv.two_way_demean(M, firm_ids, time_ids)
    y_dm = M_dm[:, 0]
    x_dm = M_dm[:, 1]
    z_dm = M_dm[:, 2]
    w_dm = M_dm[:, 3:]
    beta, _ = piv.iv_2sls_clustered(y_dm, x_dm, z_dm, w_dm, firm_ids)

    xlo = float(np.quantile(x_dm, 0.01))
    xhi = float(np.quantile(x_dm, 0.99))
    if not np.isfinite(xlo) or not np.isfinite(xhi) or xhi <= xlo:
        xlo = float(np.nanmin(x_dm))
        xhi = float(np.nanmax(x_dm))
    if xhi <= xlo:
        xhi = xlo + 1.0

    return {
        "beta": float(beta[0]),
        "xlo": xlo,
        "xhi": xhi,
        "n": int(len(x_dm)),
    }


def make_label(spec: dict) -> str:
    return (
        f"c={int(spec['cutoff'])}, bw={int(spec['bandwidth'])}, "
        f"F={spec['avg_f']:.2f}"
    )


def main():
    global rank_map

    set_research_style()
    rank_map, _ = piv.load_rank_cutoff_map(piv.RANK_PATH)
    specs = select_specs()
    labels = [make_label(s) for s in specs]

    fitted = {dep: [] for dep in piv.DEP_VARS}
    for spec in specs:
        for dep in piv.DEP_VARS:
            fitted[dep].append(
                compute_fitted(dep, spec["cutoff"], spec["bandwidth"])
            )

    # Use a restrained, colorblind-friendly palette with enough contrast
    # to distinguish the five windows when curves overlap.
    colors = ["#4e79a7", "#76b7b2", "#59a14f", "#f28e2b", "#e15759"]

    fig, axes = plt.subplots(
        len(piv.DEP_VARS),
        1,
        figsize=(9.2, 11.2),
        sharex=False,
    )
    if len(piv.DEP_VARS) == 1:
        axes = [axes]

    fig.suptitle(
        "Local-window IV fitted slopes by first-stage strength",
        fontsize=14,
        y=0.985,
    )

    handles = [
        Line2D([0], [0], color=colors[i], linewidth=2.4, label=labels[i])
        for i in range(len(labels))
    ]

    for ax, dep in zip(axes, piv.DEP_VARS):
        dep_data = fitted[dep]
        panel_xlo = min(sound["xlo"] for sound in dep_data)
        panel_xhi = max(sound["xhi"] for sound in dep_data)
        x_grid = np.linspace(panel_xlo, panel_xhi, 500)

        for color, sound in zip(colors, dep_data):
            y_line = sound["beta"] * x_grid
            mask = (x_grid >= sound["xlo"]) & (x_grid <= sound["xhi"])
            y_line = np.where(mask, y_line, np.nan)
            ax.plot(x_grid, y_line, color=color, linewidth=2.4)

        ax.axhline(0.0, color="#666666", linestyle=(0, (2, 2)), linewidth=0.9)
        ax.set_xlim(panel_xlo, panel_xhi)
        ax.set_title(f"{PANEL_MARKS[dep]} {DEP_LABELS[dep]}", loc="left", pad=6)
        ax.grid(axis="y", color="#d9d9d9", linestyle=(0, (2, 2)), linewidth=0.8)
        ax.yaxis.set_major_locator(MaxNLocator(4))
        ax.xaxis.set_major_locator(MaxNLocator(5))
        ax.tick_params(axis="both", which="major", length=4, width=0.8, color="#333333")
        ax.spines["left"].set_color("#333333")
        ax.spines["bottom"].set_color("#333333")

    axes[0].set_ylabel("Fitted 2SLS component")
    for ax in axes[:-1]:
        ax.tick_params(labelbottom=False)
    axes[-1].set_xlabel("Demeaned ind_own")

    fig.subplots_adjust(left=0.12, right=0.98, top=0.92, bottom=0.18, hspace=0.42)
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.055),
        ncol=2,
        frameon=False,
        columnspacing=1.3,
        handlelength=2.8,
    )
    fig.text(
        0.5,
        0.018,
        "Representative local windows are ordered from the weakest to the strongest first stage.",
        ha="center",
        va="bottom",
        fontsize=9,
        color="#4b5563",
    )
    fig.savefig(OUT_FIG, format="svg", dpi=240, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {OUT_FIG}")


if __name__ == "__main__":
    main()
