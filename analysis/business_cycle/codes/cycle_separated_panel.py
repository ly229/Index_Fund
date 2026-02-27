#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DATA_PATH = Path("data/df_regression.csv")
OUT_DIR = Path("analysis/business_cycle")

DEP_VARS = ["amihud_illiq", "volatility", "price_info"]
RHS_VARS = ["ind_own", "c_firm_size", "c_dollar_vol"]

# Covers full sample 1993Q1-2023Q4 without overlap.
CYCLES = [
    ("exp_1993Q1_2000Q4", "1993Q1", "2000Q4"),
    ("rec_2001Q1_2001Q4", "2001Q1", "2001Q4"),
    ("exp_2002Q1_2007Q3", "2002Q1", "2007Q3"),
    ("rec_2007Q4_2009Q2", "2007Q4", "2009Q2"),
    ("exp_2009Q3_2019Q3", "2009Q3", "2019Q3"),
    ("rec_2019Q4_2020Q2", "2019Q4", "2020Q2"),
    ("exp_2020Q3_2023Q4", "2020Q3", "2023Q4"),
]

OUT_DETAIL = OUT_DIR / "panel_cycle_separated_detailed.csv"
OUT_SUMMARY = OUT_DIR / "panel_cycle_separated_summary.csv"
OUT_TEX = OUT_DIR / "panel_cycle_separated_pretty.tex"
OUT_PLOT = OUT_DIR / "ind_own_coef_across_cycles.svg"
OUT_PLOT_TIME = OUT_DIR / "ind_own_coef_across_cycles_timeaxis.svg"


@dataclass
class FitResult:
    cycle: str
    dep_var: str
    nobs: int
    n_entities: int
    n_time: int
    coef: dict[str, float]
    se: dict[str, float]
    pval: dict[str, float]


def stars(p: float) -> str:
    if p < 0.01:
        return "***"
    if p < 0.05:
        return "**"
    if p < 0.10:
        return "*"
    return ""


def period_mask(quarters: pd.Series, q_start: str, q_end: str) -> pd.Series:
    q = quarters.astype(str)
    return (q >= q_start) & (q <= q_end)


def prep(df: pd.DataFrame, dep_var: str, q_start: str, q_end: str) -> pd.DataFrame:
    cols = ["cusip", "date", dep_var] + RHS_VARS
    d = df[cols].copy()
    d["date"] = pd.to_datetime(d["date"])
    d["quarter"] = d["date"].dt.to_period("Q")
    d = d[period_mask(d["quarter"], q_start, q_end)].copy()

    for c in [dep_var] + RHS_VARS:
        d[c] = pd.to_numeric(d[c], errors="coerce")
        d.loc[~np.isfinite(d[c]), c] = np.nan

    d = d.dropna(subset=[dep_var] + RHS_VARS)
    d = d.set_index(["cusip", "date"]).sort_index()
    return d


def run_one(df: pd.DataFrame, dep_var: str, cycle: str, q_start: str, q_end: str) -> FitResult:
    d = prep(df, dep_var, q_start, q_end)
    formula = f"{dep_var} ~ 1 + {' + '.join(RHS_VARS)} + EntityEffects + TimeEffects"
    model = PanelOLS.from_formula(formula, data=d, drop_absorbed=True)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        fit = model.fit(cov_type="clustered", cluster_entity=True, cluster_time=True)
        coef = {k: float(fit.params[k]) for k in RHS_VARS}
        se = {k: float(fit.std_errors[k]) for k in RHS_VARS}
        pval = {k: float(fit.pvalues[k]) for k in RHS_VARS}

    return FitResult(
        cycle=cycle,
        dep_var=dep_var,
        nobs=int(fit.nobs),
        n_entities=int(d.index.get_level_values(0).nunique()),
        n_time=int(d.index.get_level_values(1).nunique()),
        coef=coef,
        se=se,
        pval=pval,
    )


def build_detail(results: list[FitResult]) -> pd.DataFrame:
    rows = []
    for r in results:
        for rhs in RHS_VARS:
            rows.append(
                {
                    "cycle": r.cycle,
                    "dep_var": r.dep_var,
                    "rhs_var": rhs,
                    "coef": r.coef[rhs],
                    "se": r.se[rhs],
                    "pval": r.pval[rhs],
                    "stars": stars(r.pval[rhs]),
                    "ci95_lo": r.coef[rhs] - 1.96 * r.se[rhs],
                    "ci95_hi": r.coef[rhs] + 1.96 * r.se[rhs],
                    "nobs": r.nobs,
                    "entities": r.n_entities,
                    "time_periods": r.n_time,
                }
            )
    return pd.DataFrame(rows)


def build_summary(detail: pd.DataFrame) -> pd.DataFrame:
    sub = detail[detail["rhs_var"] == "ind_own"].copy()
    out = sub[["cycle", "dep_var", "coef", "se", "pval", "stars", "ci95_lo", "ci95_hi", "nobs", "entities", "time_periods"]]
    return out.sort_values(["cycle", "dep_var"]).reset_index(drop=True)


def pretty_cycle(c: str) -> str:
    return c.replace("_", "\n")


def plot_indown_cycle(summary: pd.DataFrame) -> None:
    order = [c[0] for c in CYCLES]
    dep_colors = {
        "amihud_illiq": "#1f77b4",
        "volatility": "#d62728",
        "price_info": "#2ca02c",
    }
    dep_labels = {
        "amihud_illiq": "Amihud illiquidity",
        "volatility": "Volatility",
        "price_info": "Price informativeness",
    }

    x = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(13.5, 5.2))

    # Shade recession-cycle categories.
    recession_cycles = {"rec_2001Q1_2001Q4", "rec_2007Q4_2009Q2", "rec_2019Q4_2020Q2"}
    for i, cyc in enumerate(order):
        if cyc in recession_cycles:
            ax.axvspan(i - 0.5, i + 0.5, color="#add8e6", alpha=0.35, lw=0)

    offsets = {"amihud_illiq": -0.18, "volatility": 0.0, "price_info": 0.18}
    for dep in DEP_VARS:
        sub = summary[summary["dep_var"] == dep].set_index("cycle").loc[order].reset_index()
        y = sub["coef"].to_numpy()
        yerr = 1.96 * sub["se"].to_numpy()
        ax.errorbar(
            x + offsets[dep],
            y,
            yerr=yerr,
            fmt="o-",
            color=dep_colors[dep],
            capsize=3,
            linewidth=1.8,
            markersize=4.8,
            label=dep_labels[dep],
        )

    ax.axhline(0, color="#666666", linestyle="--", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels([pretty_cycle(c) for c in order], fontsize=8)
    ax.set_ylabel("Coefficient on ind_own")
    ax.set_title("Cycle-Separated Panel FE: ind_own Coefficients Across Business Cycles")
    ax.grid(axis="y", alpha=0.25, linestyle=":")
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.15))
    fig.tight_layout()
    fig.savefig(OUT_PLOT, format="svg", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_indown_cycle_timeaxis(summary: pd.DataFrame) -> None:
    cycle_to_bounds = {name: (qs, qe) for name, qs, qe in CYCLES}
    full_q = pd.period_range(CYCLES[0][1], CYCLES[-1][2], freq="Q")
    x = full_q.to_timestamp(how="end")

    dep_colors = {
        "amihud_illiq": "#1f77b4",
        "volatility": "#d62728",
        "price_info": "#2ca02c",
    }
    dep_labels = {
        "amihud_illiq": "Amihud illiquidity",
        "volatility": "Volatility",
        "price_info": "Price informativeness",
    }

    fig, ax = plt.subplots(figsize=(13.5, 5.2))

    for dep in DEP_VARS:
        vals = np.full(len(full_q), np.nan)
        dep_sub = summary[summary["dep_var"] == dep].set_index("cycle")
        for cyc, (qs, qe) in cycle_to_bounds.items():
            coef = float(dep_sub.loc[cyc, "coef"])
            mask = (full_q.astype(str) >= qs) & (full_q.astype(str) <= qe)
            vals[mask] = coef
        ax.plot(
            x,
            vals,
            color=dep_colors[dep],
            linewidth=2.0,
            label=dep_labels[dep],
        )

    recession_windows = [("2001Q1", "2001Q4"), ("2007Q4", "2009Q2"), ("2019Q4", "2020Q2")]
    for i, (qs, qe) in enumerate(recession_windows):
        s = pd.Period(qs, freq="Q").to_timestamp(how="start")
        e = pd.Period(qe, freq="Q").to_timestamp(how="end")
        ax.axvspan(s, e, color="#add8e6", alpha=0.35, lw=0)
        if i == 0:
            ax.text(s, ax.get_ylim()[1], " recession", va="top", ha="left", fontsize=8, color="#2f4f4f")

    tick_q = full_q[::8]
    ax.set_xticks(tick_q.to_timestamp(how="end"))
    ax.set_xticklabels([str(q) for q in tick_q], rotation=45, ha="right", fontsize=8)
    ax.axhline(0, color="#666666", linestyle="--", linewidth=1)
    ax.set_ylabel("Coefficient on ind_own")
    ax.set_xlabel("Date (Year-Quarter)")
    ax.set_title("Cycle-Separated Panel FE: ind_own Coefficient Path with Recession Shading")
    ax.grid(axis="y", alpha=0.25, linestyle=":")
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.16))
    fig.tight_layout()
    fig.savefig(OUT_PLOT_TIME, format="svg", dpi=200, bbox_inches="tight")
    plt.close(fig)


def render_tex(summary: pd.DataFrame) -> str:
    order = [c[0] for c in CYCLES]
    dep_order = DEP_VARS
    rows = []
    rows.append("\\begin{table}[!htbp]\\centering")
    rows.append("\\caption{Cycle-Separated Panel FE: Coefficient on ind_own}")
    rows.append("\\label{tab:cycle_sep_indown}")
    rows.append("\\begin{tabular}{lccc}")
    rows.append("\\toprule")
    rows.append("Cycle & amihud_illiq & volatility & price_info \\\\")
    rows.append("\\midrule")
    for cyc in order:
        vals = []
        for dep in dep_order:
            r = summary[(summary["cycle"] == cyc) & (summary["dep_var"] == dep)].iloc[0]
            vals.append(f"{r['coef']:.4f}{r['stars']} ({r['se']:.4f})")
        rows.append(f"{cyc} & " + " & ".join(vals) + " \\\\")
    rows.append("\\bottomrule")
    rows.append("\\end{tabular}")
    rows.append("\\begin{tablenotes}\\footnotesize")
    rows.append("\\item All regressions include firm and time FE with two-way clustered SE by firm and date.")
    rows.append("\\item Recession periods follow NBER windows provided by the user.")
    rows.append("\\item Entries report coefficient on ind_own with SE in parentheses. * p$<$0.10, ** p$<$0.05, *** p$<$0.01.")
    rows.append("\\end{tablenotes}")
    rows.append("\\end{table}")
    return "\n".join(rows) + "\n"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA_PATH)

    results = []
    for cycle, qs, qe in CYCLES:
        for dep in DEP_VARS:
            results.append(run_one(df, dep, cycle, qs, qe))

    detail = build_detail(results)
    detail.to_csv(OUT_DETAIL, index=False)

    summary = build_summary(detail)
    summary.to_csv(OUT_SUMMARY, index=False)

    plot_indown_cycle(summary)
    plot_indown_cycle_timeaxis(summary)
    OUT_TEX.write_text(render_tex(summary), encoding="utf-8")

    print(f"Saved: {OUT_DETAIL}")
    print(f"Saved: {OUT_SUMMARY}")
    print(f"Saved: {OUT_PLOT}")
    print(f"Saved: {OUT_PLOT_TIME}")
    print(f"Saved: {OUT_TEX}")


if __name__ == "__main__":
    main()
