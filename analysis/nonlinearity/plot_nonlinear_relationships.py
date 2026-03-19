#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import PercentFormatter


COEF_PATH = Path("analysis/nonlinearity/nonlinear_quadratic_results.csv")
OUT_DIR = Path("analysis/nonlinearity")
OUT_PLOT = OUT_DIR / "nonlinear_relationship_all3.svg"

DEP_VARS = ["amihud_illiq", "volatility", "price_info"]
COLORS = {
    "amihud_illiq": "#1f77b4",
    "volatility": "#d62728",
    "price_info": "#2ca02c",
}
LABELS = {
    "amihud_illiq": "Amihud illiquidity",
    "volatility": "Volatility",
    "price_info": "Price informativeness",
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
            "axes.titlesize": 12,
            "axes.titleweight": "semibold",
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 9.5,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def style_axis(ax: plt.Axes) -> None:
    ax.grid(axis="y", color="#d9d9d9", linestyle=(0, (2, 2)), linewidth=0.8)
    ax.tick_params(axis="both", which="major", length=4, width=0.8, color="#333333")
    ax.spines["left"].set_color("#333333")
    ax.spines["bottom"].set_color("#333333")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    coef = pd.read_csv(COEF_PATH).set_index("dep_var")
    set_research_style()

    fig = plt.figure(figsize=(12.2, 8.6))
    panel_w = 0.31
    panel_h = 0.31
    top_y = 0.57
    bottom_y = 0.16
    left_x = 0.10
    right_x = 0.59
    center_x = 0.345
    axes = [
        fig.add_axes([left_x, top_y, panel_w, panel_h]),
        fig.add_axes([right_x, top_y, panel_w, panel_h]),
        fig.add_axes([center_x, bottom_y, panel_w, panel_h]),
    ]

    panel_marks = {"amihud_illiq": "a)", "volatility": "b)", "price_info": "c)"}
    for ax, dep in zip(axes, DEP_VARS):
        r = coef.loc[dep]
        b1 = float(r["beta_ind_own"])
        b2 = float(r["beta_ind_own_sq"])
        q01 = float(r["ind_own_q01"])
        q05 = float(r["ind_own_q05"])
        q50 = float(r["ind_own_q50"])
        q95 = float(r["ind_own_q95"])
        q99 = float(r["ind_own_q99"])
        turn = float(r["turning_point"])
        in_support = bool(r["turning_in_q01_q99"])

        x = np.linspace(q01, q99, 300)
        y = b1 * x + b2 * x**2
        y_ref = b1 * q50 + b2 * q50**2
        dy = y - y_ref

        ymin = float(np.min(dy))
        ymax = float(np.max(dy))
        pad = 0.08 * (ymax - ymin) if ymax > ymin else 0.05

        ax.axvspan(q05, q95, color="#efefef", alpha=0.9, lw=0, zorder=0)
        ax.fill_between(x, dy, ymin - pad, color=COLORS[dep], alpha=0.10, zorder=1)
        ax.plot(x, dy, color=COLORS[dep], linewidth=2.4, zorder=3)
        ax.axhline(0, color="#666666", linestyle=(0, (1, 2)), linewidth=1.0, zorder=2)
        ax.axvline(q50, color="#444444", linestyle=":", linewidth=1.1, zorder=2)
        ax.set_ylim(ymin - pad, ymax + pad)

        if np.isfinite(turn):
            style = "-" if in_support else "--"
            alpha = 0.9 if in_support else 0.5
            ax.axvline(turn, color="#7a0177", linestyle=style, linewidth=1.4, alpha=alpha, zorder=2)
            if in_support:
                y_turn = b1 * turn + b2 * turn**2 - y_ref
                ax.scatter([turn], [y_turn], color="#7a0177", s=24, zorder=4)
                ax.annotate(
                    f"{turn:.1%}",
                    xy=(turn, y_turn),
                    xytext=(8, 12),
                    textcoords="offset points",
                    fontsize=9,
                    color="#5a005a",
                    bbox={"boxstyle": "round,pad=0.18", "facecolor": "white", "edgecolor": "#d0d0d0", "linewidth": 0.7},
                )

        ax.set_title(f"{panel_marks[dep]} {LABELS[dep]}", pad=8)
        ax.set_xlabel("Index ownership share")
        ax.xaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
        style_axis(ax)

    axes[0].set_ylabel("Predicted change vs median ind_own")
    axes[2].set_ylabel("Predicted change vs median ind_own")
    legend_handles = [
        Line2D([0], [0], color="#4d4d4d", linewidth=2.2, label="Quadratic fitted change"),
        Line2D([0], [0], color="#444444", linestyle=":", linewidth=1.1, label="Median ind_own"),
        Patch(facecolor="#efefef", edgecolor="none", alpha=0.9, label="P5-P95 support"),
        Line2D([0], [0], color="#7a0177", linewidth=1.4, label="Turning point"),
    ]
    fig.suptitle("Nonlinear Relationship Between Indexing and Market Outcomes", fontsize=14, y=0.955)
    fig.legend(handles=legend_handles, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.045))
    fig.savefig(OUT_PLOT, format="svg", dpi=240, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {OUT_PLOT}")


if __name__ == "__main__":
    main()
