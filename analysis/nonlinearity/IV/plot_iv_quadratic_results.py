#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


IN_CSV = Path("analysis/nonlinearity/iv_quadratic_cutoff2000_global.csv")
OUT_DIR = Path("analysis/nonlinearity")
OUT_COEF = OUT_DIR / "iv_quadratic_coefficients_cutoff2000_global.svg"
OUT_CURVE = OUT_DIR / "iv_quadratic_curves_cutoff2000_global.svg"

DEP_ORDER = ["amihud_illiq", "volatility", "price_info"]
DEP_LABEL = {
    "amihud_illiq": "Amihud illiquidity",
    "volatility": "Volatility",
    "price_info": "Price informativeness",
}
COLORS = {"amihud_illiq": "#1f77b4", "volatility": "#d62728", "price_info": "#2ca02c"}


def stars(p: float) -> str:
    if p < 0.01:
        return "***"
    if p < 0.05:
        return "**"
    if p < 0.10:
        return "*"
    return ""


def plot_coef(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5), sharey=True)
    params = [
        ("beta_ind_own", "se_ind_own", "Linear term: b1 (ind_own)"),
        ("beta_ind_own_sq", "se_ind_own_sq", "Quadratic term: b2 (ind_own^2)"),
    ]
    y = np.arange(len(DEP_ORDER))

    for ax, (bcol, scol, title) in zip(axes, params):
        for i, dep in enumerate(DEP_ORDER):
            r = df[df["dep_var"] == dep].iloc[0]
            b = float(r[bcol])
            se = float(r[scol])
            p = float(r["p_ind_own"] if bcol == "beta_ind_own" else r["p_ind_own_sq"])
            ax.errorbar(
                x=b,
                y=i,
                xerr=1.96 * se,
                fmt="o",
                color=COLORS[dep],
                ecolor=COLORS[dep],
                capsize=3,
                markersize=5,
                linewidth=1.6,
            )
            ax.text(b, i + 0.12, stars(p), fontsize=10, ha="center", color=COLORS[dep])
        ax.axvline(0, color="#666", linestyle="--", linewidth=1)
        ax.set_title(title, fontsize=10)
        ax.grid(axis="x", alpha=0.25, linestyle=":")
        ax.set_yticks(y)
        ax.set_yticklabels([DEP_LABEL[d] for d in DEP_ORDER], fontsize=9)
        ax.invert_yaxis()

    fig.suptitle("Quadratic IV Coefficients (Cutoff=2000, Global Sample)", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT_COEF, format="svg", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_curves(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharex=False)
    for ax, dep in zip(axes, DEP_ORDER):
        r = df[df["dep_var"] == dep].iloc[0]
        b1 = float(r["beta_ind_own"])
        b2 = float(r["beta_ind_own_sq"])
        q01 = float(r["ind_own_q01"])
        q99 = float(r["ind_own_q99"])
        x = np.linspace(q01, q99, 350)
        y = b1 * x + b2 * x * x
        x_ref = 0.5 * (q01 + q99)
        y = y - (b1 * x_ref + b2 * x_ref * x_ref)

        ax.plot(x, y, color=COLORS[dep], linewidth=2.0)
        ax.axhline(0, color="#666", linestyle="--", linewidth=1)
        ax.axvspan(q01, q99, color="#d9d9d9", alpha=0.25, lw=0)
        tp = float(r["turning_point"])
        if np.isfinite(tp):
            in_support = bool(r["turning_in_q01_q99"])
            ax.axvline(tp, color="#9467bd", linestyle="-" if in_support else "--", linewidth=1.4, alpha=0.9 if in_support else 0.5)
        ax.set_title(DEP_LABEL[dep], fontsize=10)
        ax.set_xlabel("ind_own")
        ax.grid(axis="y", alpha=0.25, linestyle=":")

    axes[0].set_ylabel("Predicted IV relationship (centered)")
    fig.suptitle("Quadratic IV Fitted Curves (Cutoff=2000, Global Sample)", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT_CURVE, format="svg", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    df = pd.read_csv(IN_CSV)
    plot_coef(df)
    plot_curves(df)
    print(f"Saved: {OUT_COEF}")
    print(f"Saved: {OUT_CURVE}")


if __name__ == "__main__":
    main()
