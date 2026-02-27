#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


IN_CSV = Path("analysis/nonlinearity/iv_quadratic_cutoff2000_global.csv")
OUT_FIG = Path("analysis/nonlinearity/iv_quadratic_fitted_lines_cutoff2000_global.svg")

DEP_ORDER = ["amihud_illiq", "volatility", "price_info"]
DEP_LABEL = {
    "amihud_illiq": "Amihud illiquidity",
    "volatility": "Volatility",
    "price_info": "Price informativeness",
}
COLORS = {"amihud_illiq": "#1f77b4", "volatility": "#d62728", "price_info": "#2ca02c"}


def main() -> None:
    df = pd.read_csv(IN_CSV)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))

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

        ax.plot(x, y, color=COLORS[dep], linewidth=2.2)
        ax.set_title(DEP_LABEL[dep], fontsize=10.5)
        ax.set_xlabel("ind_own")
        ax.grid(axis="y", alpha=0.25, linestyle=":")
        ax.axvspan(q01, q99, color="#d9d9d9", alpha=0.2, lw=0)

        if np.isfinite(tp):
            ax.axvline(tp, color="#9467bd", linestyle="-" if in_support else "--", linewidth=1.4, alpha=0.9 if in_support else 0.5)

    axes[0].set_ylabel("Fitted IV component: b1*x + b2*x^2")
    fig.suptitle("Quadratic IV Fitted Lines (Global Sample, Cutoff=2000)", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT_FIG, format="svg", dpi=240, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {OUT_FIG}")


if __name__ == "__main__":
    main()
