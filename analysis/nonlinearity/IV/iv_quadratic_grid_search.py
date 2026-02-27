#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


sys.path.append("analysis/nonlinearity")
import iv_quadratic_rank_cutoff as base


OUT_DIR = Path("analysis/nonlinearity")
OUT_GRID = OUT_DIR / "iv_quadratic_grid_results.csv"
OUT_BEST = OUT_DIR / "iv_quadratic_grid_best_specs.csv"
OUT_SUMMARY = OUT_DIR / "iv_quadratic_grid_summary.md"

CUTOFFS = [1000.0, 1500.0, 2000.0, 2500.0]
BANDWIDTHS = [50.0, 100.0, 150.0, 200.0, 300.0]
DEP_VARS = ["amihud_illiq", "volatility", "price_info"]


def run_grid():
    rank_map = base.load_rank_map()
    rows = []
    for c in CUTOFFS:
        for bw in BANDWIDTHS:
            base.CUTOFF = c
            base.BANDWIDTH = bw
            for dep in DEP_VARS:
                r = base.run_one(dep, rank_map)
                r["cutoff"] = c
                r["bandwidth"] = bw
                r["min_first_stage_F"] = min(r["first_stage_jointF_ind_own"], r["first_stage_jointF_ind_own_sq"])
                rows.append(r)
    return pd.DataFrame(rows)


def plot_first_stage(df: pd.DataFrame) -> None:
    for dep in DEP_VARS:
        d = df[df["dep_var"] == dep].copy()
        fig, ax = plt.subplots(figsize=(8.5, 4.2))
        for c in CUTOFFS:
            s = d[d["cutoff"] == c].sort_values("bandwidth")
            ax.plot(s["bandwidth"], s["min_first_stage_F"], marker="o", linewidth=1.8, label=f"cutoff={int(c)}")
        ax.axhline(10.0, color="#666666", linestyle="--", linewidth=1, label="F=10")
        ax.set_title(f"Quadratic IV First-Stage Strength: {dep}")
        ax.set_xlabel("Bandwidth")
        ax.set_ylabel("min joint F across endogenous terms")
        ax.grid(axis="y", alpha=0.25, linestyle=":")
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        fig.savefig(OUT_DIR / f"iv_quadratic_grid_first_stage_{dep}.svg", format="svg", dpi=220, bbox_inches="tight")
        plt.close(fig)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = run_grid()
    df.to_csv(OUT_GRID, index=False)

    best = (
        df.sort_values(["dep_var", "min_first_stage_F"], ascending=[True, False])
        .groupby("dep_var", as_index=False)
        .head(1)
        .copy()
    )
    best.to_csv(OUT_BEST, index=False)

    plot_first_stage(df)

    lines = []
    lines.append("# Quadratic IV Robustness Grid")
    lines.append("")
    lines.append("Ranking criterion: maximize `min(F_ind_own, F_ind_own_sq)` within each dependent variable.")
    lines.append("")
    for dep in DEP_VARS:
        r = best[best["dep_var"] == dep].iloc[0]
        lines.append(f"## {dep}")
        lines.append(
            f"- Best spec: cutoff={int(r['cutoff'])}, bandwidth={int(r['bandwidth'])}, "
            f"min first-stage F={r['min_first_stage_F']:.3f}"
        )
        lines.append(
            f"- Coefficients: b1={r['beta_ind_own']:.4g} (p={r['p_ind_own']:.3g}), "
            f"b2={r['beta_ind_own_sq']:.4g} (p={r['p_ind_own_sq']:.3g})"
        )
        lines.append("")
    lines.append("Note: If best min first-stage F remains far below 10, nonlinear IV remains weakly identified.")
    OUT_SUMMARY.write_text("\n".join(lines), encoding="utf-8")

    print(f"Saved: {OUT_GRID}")
    print(f"Saved: {OUT_BEST}")
    print(f"Saved: {OUT_SUMMARY}")
    for dep in DEP_VARS:
        print(f"Saved: {OUT_DIR / f'iv_quadratic_grid_first_stage_{dep}.svg'}")


if __name__ == "__main__":
    main()
