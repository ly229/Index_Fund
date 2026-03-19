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


IN_CSV = Path("analysis/nonlinearity/iv_quadratic_cutoff2000_global.csv")
OUT_FIG = Path("analysis/nonlinearity/iv_quadratic_fitted_lines_cutoff2000_global.svg")

DEP_ORDER = ["amihud_illiq", "volatility", "price_info"]
DEP_LABEL = {
    "amihud_illiq": "Amihud illiquidity",
    "volatility": "Volatility",
    "price_info": "Price informativeness",
}
COLORS = {"amihud_illiq": "#1f77b4", "volatility": "#d62728", "price_info": "#2ca02c"}


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
            "axes.titlesize": 13,
            "axes.titleweight": "semibold",
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
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
    set_research_style()
    df = pd.read_csv(IN_CSV)
    fig = plt.figure(figsize=(13.2, 9.2))
    panel_w = 0.33
    panel_h = 0.33
    top_y = 0.55
    bottom_y = 0.15
    left_x = 0.08
    right_x = 0.59
    center_x = 0.335
    axes = [
        fig.add_axes([left_x, top_y, panel_w, panel_h]),
        fig.add_axes([right_x, top_y, panel_w, panel_h]),
        fig.add_axes([center_x, bottom_y, panel_w, panel_h]),
    ]
    panel_marks = {"amihud_illiq": "a)", "volatility": "b)", "price_info": "c)"}

    for ax, dep in zip(axes, DEP_ORDER):
        r = df[df["dep_var"] == dep].iloc[0]
        b1 = float(r["beta_ind_own"])
        b2 = float(r["beta_ind_own_sq"])
        q01 = float(r["ind_own_q01"])
        q99 = float(r["ind_own_q99"])
        tp = float(r["turning_point"])
        in_support = bool(r["turning_in_q01_q99"])

        x = np.linspace(q01, q99, 400)
        y = b1 * x + b2 * (x**2)
        ymin = float(np.min(y))
        ymax = float(np.max(y))
        pad = 0.08 * (ymax - ymin) if ymax > ymin else 0.05

        ax.axvspan(q01, q99, color="#f3f3f3", alpha=0.95, lw=0, zorder=0)
        ax.fill_between(x, y, ymin - pad, color=COLORS[dep], alpha=0.10, zorder=1)
        ax.plot(x, y, color=COLORS[dep], linewidth=2.5, zorder=3)
        ax.axhline(0.0, color="#666666", linestyle=(0, (1, 2)), linewidth=1.0, zorder=2)
        ax.set_ylim(ymin - pad, ymax + pad)
        ax.set_title(f"{panel_marks[dep]} {DEP_LABEL[dep]}", loc="center", pad=8)
        ax.set_xlabel("Index ownership share")
        ax.xaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
        style_axis(ax)

        if np.isfinite(tp):
            ax.axvline(tp, color="#7a0177", linestyle="-" if in_support else "--", linewidth=1.5, alpha=0.9 if in_support else 0.55, zorder=4)
            if in_support:
                y_tp = b1 * tp + b2 * (tp**2)
                ax.scatter([tp], [y_tp], color="#7a0177", s=26, zorder=5)
                ax.annotate(
                    f"Turning point = {tp:.3f}",
                    xy=(tp, y_tp),
                    xytext=(10, 14),
                    textcoords="offset points",
                    fontsize=9.5,
                    color="#5a005a",
                    arrowprops={"arrowstyle": "-", "color": "#5a005a", "lw": 0.8},
                )
        ax.text(
            0.03,
            0.06,
            f"Support: [{q01:.3f}, {q99:.3f}]",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=9,
            color="#4d4d4d",
        )

    axes[0].set_ylabel("Fitted IV component: $b_1 x + b_2 x^2$")
    axes[2].set_ylabel("Fitted IV component: $b_1 x + b_2 x^2$")
    legend_handles = [
        Line2D([0], [0], color="#4d4d4d", linewidth=2.2, label="Quadratic fitted line"),
        Patch(facecolor="#f3f3f3", edgecolor="none", alpha=0.95, label="Observed support"),
        Line2D([0], [0], color="#7a0177", linewidth=1.4, label="Turning point"),
    ]
    fig.suptitle("Quadratic IV Fitted Lines", fontsize=14, y=0.955)
    fig.legend(handles=legend_handles, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.045))
    fig.savefig(OUT_FIG, format="svg", dpi=240, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {OUT_FIG}")


if __name__ == "__main__":
    main()
