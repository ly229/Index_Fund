#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS


DATA_PATH = Path("data/df_regression.csv")
OUT_DIR = Path("analysis/base_penel")

DEP_VARS = ["amihud_illiq", "volatility", "price_info"]
RHS_VARS = ["ind_own", "c_firm_size", "c_dollar_vol"]

OUT_SUMMARY_CSV = OUT_DIR / "panel_results_controls.csv"
OUT_DETAIL_CSV = OUT_DIR / "panel_results_controls_detailed.csv"
OUT_LATEX = OUT_DIR / "panel_results_controls.tex"
OUT_LATEX_PRETTY = OUT_DIR / "panel_results_controls_pretty.tex"


@dataclass
class ModelResult:
    dep_var: str
    nobs: int
    n_entities: int
    n_time: int
    coefs: dict[str, float]
    ses: dict[str, float]
    pvals: dict[str, float]


def stars(p: float) -> str:
    if p < 0.01:
        return "***"
    if p < 0.05:
        return "**"
    if p < 0.10:
        return "*"
    return ""


def run_one(df: pd.DataFrame, dep_var: str) -> ModelResult:
    use_cols = ["cusip", "date", dep_var] + RHS_VARS
    d = df[use_cols].copy()
    for c in [dep_var] + RHS_VARS:
        d[c] = pd.to_numeric(d[c], errors="coerce")
        d.loc[~np.isfinite(d[c]), c] = np.nan
    d["date"] = pd.to_datetime(d["date"])
    d = d.dropna(subset=[dep_var] + RHS_VARS)
    d = d.set_index(["cusip", "date"]).sort_index()

    formula = f"{dep_var} ~ 1 + {' + '.join(RHS_VARS)} + EntityEffects + TimeEffects"
    model = PanelOLS.from_formula(formula, data=d, drop_absorbed=True)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        fit = model.fit(cov_type="clustered", cluster_entity=True, cluster_time=True)
        coefs = {k: float(fit.params[k]) for k in RHS_VARS}
        ses = {k: float(fit.std_errors[k]) for k in RHS_VARS}
        pvals = {k: float(fit.pvalues[k]) for k in RHS_VARS}

    entities = d.index.get_level_values(0).nunique()
    times = d.index.get_level_values(1).nunique()
    return ModelResult(
        dep_var=dep_var,
        nobs=int(fit.nobs),
        n_entities=int(entities),
        n_time=int(times),
        coefs=coefs,
        ses=ses,
        pvals=pvals,
    )


def build_summary(results: list[ModelResult]) -> pd.DataFrame:
    table = pd.DataFrame(index=RHS_VARS + ["N", "Entities", "Time", "FE", "SE"], columns=DEP_VARS)
    for r in results:
        for v in RHS_VARS:
            table.loc[v, r.dep_var] = f"{r.coefs[v]:.4f} ({r.ses[v]:.4f})"
        table.loc["N", r.dep_var] = str(r.nobs)
        table.loc["Entities", r.dep_var] = str(r.n_entities)
        table.loc["Time", r.dep_var] = str(r.n_time)
        table.loc["FE", r.dep_var] = "Entity+Time"
        table.loc["SE", r.dep_var] = "Two-way clustered"
    return table


def build_detail(results: list[ModelResult]) -> pd.DataFrame:
    rows = []
    for r in results:
        for v in RHS_VARS:
            rows.append(
                {
                    "dep_var": r.dep_var,
                    "rhs_var": v,
                    "coef": r.coefs[v],
                    "se": r.ses[v],
                    "pval": r.pvals[v],
                    "stars": stars(r.pvals[v]),
                    "nobs": r.nobs,
                    "entities": r.n_entities,
                    "time_periods": r.n_time,
                    "fe": "Entity+Time",
                    "se_type": "Two-way clustered",
                }
            )
    return pd.DataFrame(rows)


def render_pretty_tex(results: list[ModelResult]) -> str:
    labels = {
        "ind_own": "Industry ownership (ind_own)",
        "c_firm_size": "Firm size (c_firm_size)",
        "c_dollar_vol": "Dollar volume (c_dollar_vol)",
    }
    lines: list[str] = []
    lines.append("\\begin{table}[!htbp]\\centering")
    lines.append("\\caption{Panel Regressions with Firm and Time Fixed Effects}")
    lines.append("\\label{tab:panel_controls}")
    lines.append("\\begin{tabular}{lccc}")
    lines.append("\\toprule")
    lines.append(" & amihud_illiq & volatility & price_info \\\\")
    lines.append("\\midrule")

    by_dep = {r.dep_var: r for r in results}
    for var in RHS_VARS:
        row_coef = []
        row_se = []
        for dep in DEP_VARS:
            r = by_dep[dep]
            row_coef.append(f"{r.coefs[var]:.4f}{stars(r.pvals[var])}")
            row_se.append(f"({r.ses[var]:.4f})")
        lines.append(f"{labels[var]} & " + " & ".join(row_coef) + " \\\\")
        lines.append(" & " + " & ".join(row_se) + " \\\\")

    lines.append("\\midrule")
    lines.append("Observations & " + " & ".join(str(by_dep[d].nobs) for d in DEP_VARS) + " \\\\")
    lines.append("Entities & " + " & ".join(str(by_dep[d].n_entities) for d in DEP_VARS) + " \\\\")
    lines.append("Time periods & " + " & ".join(str(by_dep[d].n_time) for d in DEP_VARS) + " \\\\")
    lines.append("Firm FE & Yes & Yes & Yes \\\\")
    lines.append("Time FE & Yes & Yes & Yes \\\\")
    lines.append("Two-way clustered SE & Yes & Yes & Yes \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\begin{tablenotes}\\footnotesize")
    lines.append("\\item Notes: Two-way clustered standard errors by firm and date in parentheses. * p$<$0.10, ** p$<$0.05, *** p$<$0.01.")
    lines.append("\\end{tablenotes}")
    lines.append("\\end{table}")
    return "\n".join(lines) + "\n"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA_PATH)
    results = [run_one(df, dep) for dep in DEP_VARS]

    summary = build_summary(results)
    summary.to_csv(OUT_SUMMARY_CSV)

    detail = build_detail(results)
    detail.to_csv(OUT_DETAIL_CSV, index=False)

    summary.to_latex(OUT_LATEX, index=True)
    OUT_LATEX_PRETTY.write_text(render_pretty_tex(results), encoding="utf-8")

    print(f"Saved: {OUT_SUMMARY_CSV}")
    print(f"Saved: {OUT_DETAIL_CSV}")
    print(f"Saved: {OUT_LATEX}")
    print(f"Saved: {OUT_LATEX_PRETTY}")


if __name__ == "__main__":
    main()
