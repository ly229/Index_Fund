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
BASE_RHS = ["ind_own", "c_firm_size", "c_dollar_vol"]
INT_VAR = "ind_own_x_recession"

OUT_DETAIL = OUT_DIR / "panel_recession_interaction_detailed.csv"
OUT_SUMMARY = OUT_DIR / "panel_recession_interaction_summary.csv"
OUT_TEX = OUT_DIR / "panel_recession_interaction_pretty.tex"


RECESSION_QUARTERS = {
    "2001Q1",
    "2001Q2",
    "2001Q3",
    "2001Q4",
    "2007Q4",
    "2008Q1",
    "2008Q2",
    "2008Q3",
    "2008Q4",
    "2009Q1",
    "2009Q2",
    "2019Q4",
    "2020Q1",
    "2020Q2",
}


@dataclass
class ModelResult:
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


def prep_data(df: pd.DataFrame, dep_var: str) -> pd.DataFrame:
    cols = ["cusip", "date", dep_var] + BASE_RHS
    d = df[cols].copy()
    d["date"] = pd.to_datetime(d["date"])
    d["quarter"] = d["date"].dt.to_period("Q").astype(str)
    d["recession"] = d["quarter"].isin(RECESSION_QUARTERS).astype(float)
    d[INT_VAR] = d["ind_own"] * d["recession"]

    for c in [dep_var] + BASE_RHS + [INT_VAR]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
        d.loc[~np.isfinite(d[c]), c] = np.nan

    d = d.dropna(subset=[dep_var] + BASE_RHS + [INT_VAR])
    d = d.set_index(["cusip", "date"]).sort_index()
    return d


def run_one(df: pd.DataFrame, dep_var: str) -> ModelResult:
    d = prep_data(df, dep_var)

    rhs = BASE_RHS + [INT_VAR]
    formula = f"{dep_var} ~ 1 + {' + '.join(rhs)} + EntityEffects + TimeEffects"
    model = PanelOLS.from_formula(formula, data=d, drop_absorbed=True)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        fit = model.fit(cov_type="clustered", cluster_entity=True, cluster_time=True)
        coef = {k: float(fit.params[k]) for k in rhs}
        se = {k: float(fit.std_errors[k]) for k in rhs}
        pval = {k: float(fit.pvalues[k]) for k in rhs}

    entities = d.index.get_level_values(0).nunique()
    times = d.index.get_level_values(1).nunique()
    return ModelResult(
        dep_var=dep_var,
        nobs=int(fit.nobs),
        n_entities=int(entities),
        n_time=int(times),
        coef=coef,
        se=se,
        pval=pval,
    )


def to_detail(results: list[ModelResult]) -> pd.DataFrame:
    rows = []
    rhs = BASE_RHS + [INT_VAR]
    for r in results:
        for v in rhs:
            rows.append(
                {
                    "dep_var": r.dep_var,
                    "rhs_var": v,
                    "coef": r.coef[v],
                    "se": r.se[v],
                    "pval": r.pval[v],
                    "stars": stars(r.pval[v]),
                    "nobs": r.nobs,
                    "entities": r.n_entities,
                    "time_periods": r.n_time,
                }
            )
    return pd.DataFrame(rows)


def to_summary(results: list[ModelResult]) -> pd.DataFrame:
    idx = ["ind_own", "ind_own_x_recession", "c_firm_size", "c_dollar_vol", "N", "Entities", "Time", "FE", "SE"]
    tbl = pd.DataFrame(index=idx, columns=DEP_VARS)
    for r in results:
        tbl.loc["ind_own", r.dep_var] = f"{r.coef['ind_own']:.4f} ({r.se['ind_own']:.4f}){stars(r.pval['ind_own'])}"
        tbl.loc["ind_own_x_recession", r.dep_var] = (
            f"{r.coef[INT_VAR]:.4f} ({r.se[INT_VAR]:.4f}){stars(r.pval[INT_VAR])}"
        )
        tbl.loc["c_firm_size", r.dep_var] = (
            f"{r.coef['c_firm_size']:.4f} ({r.se['c_firm_size']:.4f}){stars(r.pval['c_firm_size'])}"
        )
        tbl.loc["c_dollar_vol", r.dep_var] = (
            f"{r.coef['c_dollar_vol']:.4f} ({r.se['c_dollar_vol']:.4f}){stars(r.pval['c_dollar_vol'])}"
        )
        tbl.loc["N", r.dep_var] = str(r.nobs)
        tbl.loc["Entities", r.dep_var] = str(r.n_entities)
        tbl.loc["Time", r.dep_var] = str(r.n_time)
        tbl.loc["FE", r.dep_var] = "Entity+Time"
        tbl.loc["SE", r.dep_var] = "Two-way clustered"
    return tbl


def render_tex(results: list[ModelResult]) -> str:
    by_dep = {r.dep_var: r for r in results}
    rows = []
    rows.append("\\begin{table}[!htbp]\\centering")
    rows.append("\\caption{Heterogeneity by Recession: Panel FE with Interaction}")
    rows.append("\\label{tab:recession_interaction}")
    rows.append("\\begin{tabular}{lccc}")
    rows.append("\\toprule")
    rows.append(" & amihud_illiq & volatility & price_info \\\\")
    rows.append("\\midrule")
    for v, label in [
        ("ind_own", "Industry ownership (ind_own)"),
        (INT_VAR, "Interaction: ind_own $\\times$ recession"),
        ("c_firm_size", "Firm size (c_firm_size)"),
        ("c_dollar_vol", "Dollar volume (c_dollar_vol)"),
    ]:
        coef_row = []
        se_row = []
        for dep in DEP_VARS:
            r = by_dep[dep]
            coef_row.append(f"{r.coef[v]:.4f}{stars(r.pval[v])}")
            se_row.append(f"({r.se[v]:.4f})")
        rows.append(f"{label} & " + " & ".join(coef_row) + " \\\\")
        rows.append(" & " + " & ".join(se_row) + " \\\\")
    rows.append("\\midrule")
    rows.append("Observations & " + " & ".join(str(by_dep[d].nobs) for d in DEP_VARS) + " \\\\")
    rows.append("Entities & " + " & ".join(str(by_dep[d].n_entities) for d in DEP_VARS) + " \\\\")
    rows.append("Time periods & " + " & ".join(str(by_dep[d].n_time) for d in DEP_VARS) + " \\\\")
    rows.append("Firm FE & Yes & Yes & Yes \\\\")
    rows.append("Time FE & Yes & Yes & Yes \\\\")
    rows.append("Two-way clustered SE & Yes & Yes & Yes \\\\")
    rows.append("\\bottomrule")
    rows.append("\\end{tabular}")
    rows.append("\\begin{tablenotes}\\footnotesize")
    rows.append("\\item Recession quarters: 2001Q1-2001Q4, 2007Q4-2009Q2, 2019Q4-2020Q2.")
    rows.append("\\item Interaction coefficient is incremental slope in recession periods relative to non-recession quarters.")
    rows.append("\\item Two-way clustered SE by firm and date in parentheses. * p$<$0.10, ** p$<$0.05, *** p$<$0.01.")
    rows.append("\\end{tablenotes}")
    rows.append("\\end{table}")
    return "\n".join(rows) + "\n"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA_PATH)
    results = [run_one(df, dep) for dep in DEP_VARS]

    detail = to_detail(results)
    detail.to_csv(OUT_DETAIL, index=False)

    summary = to_summary(results)
    summary.to_csv(OUT_SUMMARY)

    OUT_TEX.write_text(render_tex(results), encoding="utf-8")

    print(f"Saved: {OUT_DETAIL}")
    print(f"Saved: {OUT_SUMMARY}")
    print(f"Saved: {OUT_TEX}")


if __name__ == "__main__":
    main()
