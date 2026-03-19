#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import pandas as pd


INPUT_CSV = Path("analysis/nonlinearity/spline/spline_panel_results.csv")
OUTPUT_TEX = Path("analysis/nonlinearity/spline/spline_panel_results_pretty.tex")

DEP_ORDER = ["amihud_illiq", "volatility", "price_info"]
DEP_LABELS = {
    "amihud_illiq": "Amihud illiquidity",
    "volatility": "Volatility",
    "price_info": "Price informativeness",
}

SPLINE_ORDER = [1, 2, 3, 4]
SPLINE_LABELS = {i: f"Spline basis {i}" for i in SPLINE_ORDER}


def stars_from_p(p: float) -> str:
    if pd.isna(p):
        return ""
    if p < 0.01:
        return "***"
    if p < 0.05:
        return "**"
    if p < 0.10:
        return "*"
    return ""


def fmt_cell(beta: float, se: float, pval: float) -> str:
    return f"\\shortstack{{{beta:.4f}{stars_from_p(pval)}\\\\({se:.4f})}}"


def load_data() -> pd.DataFrame:
    df = pd.read_csv(INPUT_CSV)
    numeric_cols = [
        "nobs",
        "entities",
        "time_periods",
        "ind_own_min",
        "ind_own_q01",
        "ind_own_q05",
        "ind_own_q10",
        "ind_own_q50",
        "ind_own_q90",
        "ind_own_q95",
        "ind_own_q99",
        "ind_own_max",
        "best_ind_own_for_depvar",
        "best_curve_value",
        "rsquared_within",
    ]
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    for i in SPLINE_ORDER:
        for prefix in ["beta", "se", "p"]:
            df[f"{prefix}_spline_{i}"] = pd.to_numeric(df[f"{prefix}_spline_{i}"], errors="coerce")
    df["dep_var"] = pd.Categorical(df["dep_var"], categories=DEP_ORDER, ordered=True)
    return df.sort_values("dep_var").reset_index(drop=True)


def pick_row(df: pd.DataFrame, dep_var: str) -> pd.Series:
    rows = df[df["dep_var"] == dep_var]
    if rows.empty:
        raise KeyError(f"Missing row for dep_var={dep_var}")
    return rows.iloc[0]


def fmt_num(x: float) -> str:
    return f"{x:.4f}"


def build_table(df: pd.DataFrame) -> str:
    rows: list[str] = []
    rows.append("\\begin{table}[!htbp]\\centering")
    rows.append("\\caption{Spline Panel Regression Results}")
    rows.append("\\label{tab:spline_panel_results}")
    rows.append("\\begin{threeparttable}")
    rows.append("\\footnotesize")
    rows.append("\\setlength{\\tabcolsep}{3.5pt}")
    rows.append("\\renewcommand{\\arraystretch}{1.05}")
    rows.append("\\begin{tabular}{lccc}")
    rows.append("\\toprule")
    rows.append(" & " + " & ".join(DEP_LABELS[d] for d in DEP_ORDER) + " \\\\")
    rows.append("\\midrule")

    for i in SPLINE_ORDER:
        cells = []
        for dep in DEP_ORDER:
            r = pick_row(df, dep)
            cells.append(fmt_cell(float(r[f"beta_spline_{i}"]), float(r[f"se_spline_{i}"]), float(r[f"p_spline_{i}"])))
        rows.append(f"{SPLINE_LABELS[i]} & " + " & ".join(cells) + " \\\\")

    rows.append("\\midrule")
    summary_rows = [
        ("Observations", "nobs"),
        ("Entities", "entities"),
        ("Time periods", "time_periods"),
        ("Within $R^2$", "rsquared_within"),
        ("Best `ind_own`", "best_ind_own_for_depvar"),
        ("Best curve value", "best_curve_value"),
        ("Support q01", "ind_own_q01"),
        ("Support q50", "ind_own_q50"),
        ("Support q99", "ind_own_q99"),
    ]
    for label, col in summary_rows:
        values = []
        for dep in DEP_ORDER:
            r = pick_row(df, dep)
            val = r[col]
            if col in {"nobs", "entities", "time_periods"}:
                values.append(str(int(val)))
            else:
                values.append(fmt_num(float(val)))
        rows.append(label + " & " + " & ".join(values) + " \\\\")

    rows.append("\\bottomrule")
    rows.append("\\end{tabular}")
    rows.append("\\vspace{0.35em}")
    rows.append("\\noindent\\parbox{\\textwidth}{\\footnotesize")
    rows.append(
        "Notes: This table reports spline panel regressions with firm and time fixed effects. "
        "Standard errors are reported underneath coefficients in parentheses. "
        "The spline basis is a natural cubic spline in `ind_own` with four basis terms. "
        "* p$<$0.10, ** p$<$0.05, *** p$<$0.01.\\\\"
    )
    rows.append(
        "The sample support rows report the empirical ownership distribution used for interpreting the fitted spline."
    )
    rows.append("}")
    rows.append("\\end{threeparttable}")
    rows.append("\\end{table}")
    return "\n".join(rows) + "\n"


def main() -> None:
    df = load_data()
    OUTPUT_TEX.write_text(build_table(df), encoding="utf-8")
    print(f"Saved: {OUTPUT_TEX}")


if __name__ == "__main__":
    main()
