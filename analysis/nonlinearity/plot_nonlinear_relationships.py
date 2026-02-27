#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


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


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    coef = pd.read_csv(COEF_PATH).set_index("dep_var")

    fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.8), sharex=False)
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

        ax.plot(x, dy, color=COLORS[dep], linewidth=2.2, label="Quadratic fitted change")
        ax.axhline(0, color="#666666", linestyle="--", linewidth=1)
        ax.axvline(q50, color="#444444", linestyle=":", linewidth=1.2, label="Median ind_own")
        ax.axvspan(q05, q95, color="#d9d9d9", alpha=0.35, lw=0, label="P5-P95 support")

        if np.isfinite(turn):
            style = "-" if in_support else "--"
            alpha = 0.9 if in_support else 0.5
            ax.axvline(turn, color="#9467bd", linestyle=style, linewidth=1.5, alpha=alpha)
            ax.text(
                turn,
                ax.get_ylim()[0] + 0.08 * (ax.get_ylim()[1] - ax.get_ylim()[0]),
                f"turn={turn:.3f}",
                rotation=90,
                va="bottom",
                ha="right",
                fontsize=7.5,
                color="#6a3d9a",
                alpha=alpha,
            )

        ax.set_title(LABELS[dep], fontsize=10.5)
        ax.set_xlabel("ind_own")
        ax.grid(axis="y", linestyle=":", alpha=0.25)

        note = (
            f"b1={b1:.3g}, b2={b2:.3g}\n"
            f"p(b1)={float(r['p_ind_own']):.2g}, p(b2)={float(r['p_ind_own_sq']):.2g}"
        )
        ax.text(
            0.02,
            0.96,
            note,
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=8,
            bbox=dict(facecolor="white", alpha=0.8, edgecolor="#dddddd", pad=2),
        )

    axes[0].set_ylabel("Predicted change vs median ind_own")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.03))
    fig.suptitle("Nonlinear Relationship Between Indexing and Market Outcomes", fontsize=12, y=1.08)
    fig.tight_layout()
    fig.savefig(OUT_PLOT, format="svg", dpi=240, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {OUT_PLOT}")


if __name__ == "__main__":
    main()
