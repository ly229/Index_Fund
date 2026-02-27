#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


BASE_DIR = Path("analysis/base_penel")
IN_DETAIL = BASE_DIR / "panel_results_controls_detailed.csv"

OUT_STATS_CSV = BASE_DIR / "panel_results_inference_stats.csv"
OUT_STATS_TEX = BASE_DIR / "panel_results_inference_stats.tex"
OUT_INTERPRET_MD = BASE_DIR / "baseline_interpretation.md"
OUT_PLOT_SVG = BASE_DIR / "panel_coefficients_95ci.svg"

RHS_ORDER = ["ind_own", "c_firm_size", "c_dollar_vol"]
DEP_ORDER = ["amihud_illiq", "volatility", "price_info"]


def pretty_dep(dep: str) -> str:
    return {
        "amihud_illiq": "Amihud illiquidity",
        "volatility": "Volatility",
        "price_info": "Price informativeness",
    }.get(dep, dep)


def pretty_rhs(rhs: str) -> str:
    return {
        "ind_own": "Industry ownership (ind_own)",
        "c_firm_size": "Firm size (c_firm_size)",
        "c_dollar_vol": "Dollar volume (c_dollar_vol)",
    }.get(rhs, rhs)


def sig_label(p: float) -> str:
    if p < 0.01:
        return "1%"
    if p < 0.05:
        return "5%"
    if p < 0.10:
        return "10%"
    return "n.s."


def build_stats(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["t_stat"] = d["coef"] / d["se"]
    d["ci95_lo"] = d["coef"] - 1.96 * d["se"]
    d["ci95_hi"] = d["coef"] + 1.96 * d["se"]
    d["sig_level"] = d["pval"].map(sig_label)
    d["dep_label"] = d["dep_var"].map(pretty_dep)
    d["rhs_label"] = d["rhs_var"].map(pretty_rhs)
    d = d[
        [
            "dep_var",
            "dep_label",
            "rhs_var",
            "rhs_label",
            "coef",
            "se",
            "t_stat",
            "pval",
            "ci95_lo",
            "ci95_hi",
            "sig_level",
            "stars",
            "nobs",
            "entities",
            "time_periods",
            "fe",
            "se_type",
        ]
    ].copy()
    d["dep_var"] = pd.Categorical(d["dep_var"], categories=DEP_ORDER, ordered=True)
    d["rhs_var"] = pd.Categorical(d["rhs_var"], categories=RHS_ORDER, ordered=True)
    d = d.sort_values(["rhs_var", "dep_var"]).reset_index(drop=True)
    return d


def export_latex(stats: pd.DataFrame) -> None:
    out = stats.copy()
    for c in ["coef", "se", "t_stat", "pval", "ci95_lo", "ci95_hi"]:
        out[c] = out[c].map(lambda x: f"{x:.4f}")
    out = out.rename(
        columns={
            "dep_label": "Dependent variable",
            "rhs_label": "Regressor",
            "coef": "Coef.",
            "se": "SE",
            "t_stat": "t-stat",
            "pval": "p-value",
            "ci95_lo": "CI 95% Low",
            "ci95_hi": "CI 95% High",
            "sig_level": "Significance",
            "nobs": "N",
            "entities": "Entities",
            "time_periods": "Time periods",
            "fe": "Fixed effects",
            "se_type": "SE type",
        }
    )
    keep_cols = [
        "Dependent variable",
        "Regressor",
        "Coef.",
        "SE",
        "t-stat",
        "p-value",
        "CI 95% Low",
        "CI 95% High",
        "Significance",
        "N",
        "Entities",
        "Time periods",
        "Fixed effects",
        "SE type",
    ]
    out[keep_cols].to_latex(OUT_STATS_TEX, index=False, escape=True)


def render_plot(stats: pd.DataFrame) -> None:
    colors = {
        "amihud_illiq": "#1f77b4",
        "volatility": "#d62728",
        "price_info": "#2ca02c",
    }
    rhs_panels = RHS_ORDER
    fig, axes = plt.subplots(1, len(rhs_panels), figsize=(14, 4.6), sharex=False, sharey=True)
    if len(rhs_panels) == 1:
        axes = [axes]

    y_pos = np.arange(len(DEP_ORDER))
    for ax, rhs in zip(axes, rhs_panels):
        sub = stats[stats["rhs_var"] == rhs].set_index("dep_var").loc[DEP_ORDER].reset_index()
        for i, dep in enumerate(DEP_ORDER):
            row = sub[sub["dep_var"] == dep].iloc[0]
            ax.errorbar(
                x=row["coef"],
                y=i,
                xerr=1.96 * row["se"],
                fmt="o",
                color=colors[dep],
                ecolor=colors[dep],
                capsize=3,
                markersize=5,
                elinewidth=1.5,
            )
        ax.axvline(0, color="#666666", linestyle="--", linewidth=1)
        ax.set_title(pretty_rhs(rhs), fontsize=10)
        ax.set_yticks(y_pos)
        ax.set_yticklabels([pretty_dep(d) for d in DEP_ORDER], fontsize=9)
        ax.grid(axis="x", alpha=0.25, linestyle=":")
        ax.invert_yaxis()
        ax.tick_params(axis="x", labelsize=9)

    fig.suptitle("Baseline Panel OLS Coefficients with 95% CI", fontsize=12, y=1.02)
    fig.text(0.5, -0.02, "Coefficient estimate", ha="center", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT_PLOT_SVG, format="svg", dpi=200, bbox_inches="tight")
    plt.close(fig)


def write_interpretation(stats: pd.DataFrame) -> None:
    lines = []
    lines.append("# Baseline Panel OLS Interpretation")
    lines.append("")
    lines.append("Specification: firm and time fixed effects with two-way clustered SE (firm, date).")
    lines.append("")
    lines.append("## Main variable: `ind_own`")
    for dep in DEP_ORDER:
        row = stats[(stats["rhs_var"] == "ind_own") & (stats["dep_var"] == dep)].iloc[0]
        direction = "positive" if row["coef"] > 0 else "negative"
        lines.append(
            f"- {pretty_dep(dep)}: {direction} association, coef={row['coef']:.4f}, "
            f"SE={row['se']:.4f}, t={row['t_stat']:.2f}, p={row['pval']:.4g}, "
            f"95% CI=[{row['ci95_lo']:.4f}, {row['ci95_hi']:.4f}] ({row['sig_level']})."
        )
    lines.append("")
    lines.append("## Controls")
    for rhs in ["c_firm_size", "c_dollar_vol"]:
        lines.append(f"- {pretty_rhs(rhs)}")
        for dep in DEP_ORDER:
            row = stats[(stats["rhs_var"] == rhs) & (stats["dep_var"] == dep)].iloc[0]
            direction = "positive" if row["coef"] > 0 else "negative"
            lines.append(
                f"  - {pretty_dep(dep)}: {direction}, coef={row['coef']:.4f}, p={row['pval']:.4g}."
            )
    lines.append("")
    lines.append("## Sample size")
    for dep in DEP_ORDER:
        row = stats[(stats["rhs_var"] == "ind_own") & (stats["dep_var"] == dep)].iloc[0]
        lines.append(
            f"- {pretty_dep(dep)}: N={int(row['nobs']):,}, firms={int(row['entities']):,}, periods={int(row['time_periods'])}."
        )
    OUT_INTERPRET_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    df = pd.read_csv(IN_DETAIL)
    stats = build_stats(df)
    stats.to_csv(OUT_STATS_CSV, index=False)
    export_latex(stats)
    render_plot(stats)
    write_interpretation(stats)
    print(f"Saved: {OUT_STATS_CSV}")
    print(f"Saved: {OUT_STATS_TEX}")
    print(f"Saved: {OUT_PLOT_SVG}")
    print(f"Saved: {OUT_INTERPRET_MD}")


if __name__ == "__main__":
    main()
