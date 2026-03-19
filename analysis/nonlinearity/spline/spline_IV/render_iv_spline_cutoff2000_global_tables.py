#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import pandas as pd


INPUT_CSV = Path("analysis/nonlinearity/spline/spline_IV/iv_spline_cutoff2000_global_results.csv")
OUT_FIRST_STAGE_TEX = Path("analysis/nonlinearity/spline/spline_IV/iv_spline_cutoff2000_global_first_stage_pretty.tex")
OUT_IV_TEX = Path("analysis/nonlinearity/spline/spline_IV/iv_spline_cutoff2000_global_iv_pretty.tex")

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


def fmt_p(p: float) -> str:
    if pd.isna(p):
        return ""
    if p <= 0:
        return "<1e-16"
    if p < 0.0001:
        return f"{p:.2e}"
    return f"{p:.4f}"


def fmt_first_stage_cell(f_stat: float, pval: float) -> str:
    return f"\\shortstack{{{f_stat:.4f}\\\\(p={fmt_p(pval)})}}"


def fmt_iv_cell(beta: float, se: float, pval: float) -> str:
    return f"\\shortstack{{{beta:.4f}{stars_from_p(pval)}\\\\({se:.4f})}}"


def load_data() -> pd.DataFrame:
    df = pd.read_csv(INPUT_CSV)
    numeric_cols = [
        "nobs",
        "n_firms",
        "n_dates",
        "cutoff",
        "bandwidth",
        "poly_deg",
        "scale_lo",
        "scale_hi",
        "ind_own_q01",
        "ind_own_q50",
        "ind_own_q99",
        "best_ind_own_for_depvar",
        "best_curve_value",
        "knot_1",
        "knot_2",
        "knot_3",
        "min_first_stage_F",
    ]
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    for i in SPLINE_ORDER:
        for prefix in ["beta", "se", "t", "p", "first_stage_F", "first_stage_P"]:
            col = f"{prefix}_spline_{i}"
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["dep_var"] = pd.Categorical(df["dep_var"], categories=DEP_ORDER, ordered=True)
    return df.sort_values("dep_var").reset_index(drop=True)


def pick_row(df: pd.DataFrame, dep_var: str) -> pd.Series:
    rows = df[df["dep_var"] == dep_var]
    if rows.empty:
        raise KeyError(f"Missing row for dep_var={dep_var}")
    return rows.iloc[0]


def fmt_support_value(x: float) -> str:
    return f"{x:.4f}"


def fmt_range(x: float) -> str:
    return f"{x:.1%}"


def build_first_stage_table(df: pd.DataFrame) -> str:
    rows: list[str] = []
    rows.append("\\begin{table}[!htbp]\\centering")
    rows.append("\\caption{First-Stage F Statistics for Spline IV at Global Russell 2000 Cutoff}")
    rows.append("\\label{tab:iv_spline_global_first_stage}")
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
            cells.append(fmt_first_stage_cell(float(r[f"first_stage_F_spline_{i}"]), float(r[f"first_stage_P_spline_{i}"])))
        rows.append(f"{SPLINE_LABELS[i]} & " + " & ".join(cells) + " \\\\")

    rows.append("\\midrule")
    summary_rows = [
        ("Observations", "nobs"),
        ("Firms", "n_firms"),
        ("Dates", "n_dates"),
        ("Cutoff", "cutoff"),
        ("Poly degree", "poly_deg"),
        ("Min first-stage F", "min_first_stage_F"),
    ]
    for label, col in summary_rows:
        values = []
        for dep in DEP_ORDER:
            r = pick_row(df, dep)
            val = r[col]
            if col in {"nobs", "n_firms", "n_dates", "poly_deg"}:
                values.append(str(int(val)))
            elif col == "cutoff":
                values.append(f"{float(val):.0f}")
            else:
                values.append(f"{float(val):.4f}")
        rows.append(label + " & " + " & ".join(values) + " \\\\")
    rows.append("Controls & " + " & ".join(["c_firm_size, c_dollar_vol, running powers 1-4"] * len(DEP_ORDER)) + " \\\\")
    rows.append("Excluded instruments & " + " & ".join(["z_running powers 0-4"] * len(DEP_ORDER)) + " \\\\")
    rows.append("Fixed effects & " + " & ".join(["firm + date"] * len(DEP_ORDER)) + " \\\\")
    rows.append("SE type & " + " & ".join(["firm clustered"] * len(DEP_ORDER)) + " \\\\")
    rows.append("\\bottomrule")
    rows.append("\\end{tabular}")
    rows.append("\\vspace{0.35em}")
    rows.append("\\noindent\\parbox{\\textwidth}{\\footnotesize")
    rows.append(
        "Notes: This table reports first-stage relevance diagnostics for the spline IV specification. "
        "Entries are first-stage F statistics for each endogenous spline basis term, with p-values shown underneath. "
        "The model uses firm and date fixed effects and firm-clustered standard errors. "
        "* p$<$0.10, ** p$<$0.05, *** p$<$0.01.\\\\"
    )
    rows.append(
        "The spline basis is a natural cubic spline in `ind_own` with four endogenous terms and excluded instruments formed by cutoff interactions with running-variable powers."
    )
    rows.append("}")
    rows.append("\\end{threeparttable}")
    rows.append("\\end{table}")
    return "\n".join(rows) + "\n"


def build_iv_table(df: pd.DataFrame) -> str:
    rows: list[str] = []
    rows.append("\\begin{table}[!htbp]\\centering")
    rows.append("\\caption{IV Regression Results for Spline IV at Global Russell 2000 Cutoff}")
    rows.append("\\label{tab:iv_spline_global_results}")
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
            cells.append(fmt_iv_cell(float(r[f"beta_spline_{i}"]), float(r[f"se_spline_{i}"]), float(r[f"p_spline_{i}"])))
        rows.append(f"{SPLINE_LABELS[i]} & " + " & ".join(cells) + " \\\\")

    rows.append("\\midrule")
    summary_rows = [
        ("Observations", "nobs"),
        ("Firms", "n_firms"),
        ("Dates", "n_dates"),
        ("Cutoff", "cutoff"),
        ("Poly degree", "poly_deg"),
        ("Min first-stage F", "min_first_stage_F"),
        ("Best `ind_own`", "best_ind_own_for_depvar"),
    ]
    for label, col in summary_rows:
        values = []
        for dep in DEP_ORDER:
            r = pick_row(df, dep)
            val = r[col]
            if col in {"nobs", "n_firms", "n_dates", "poly_deg"}:
                values.append(str(int(val)))
            elif col == "cutoff":
                values.append(f"{float(val):.0f}")
            elif col in {"min_first_stage_F", "best_curve_value"}:
                values.append(f"{float(val):.4f}")
            else:
                values.append(fmt_range(float(val)))
        rows.append(label + " & " + " & ".join(values) + " \\\\")
    rows.append("Controls & " + " & ".join(["c_firm_size, c_dollar_vol, running powers 1-4"] * len(DEP_ORDER)) + " \\\\")
    rows.append("Excluded instruments & " + " & ".join(["z_running powers 0-4"] * len(DEP_ORDER)) + " \\\\")
    rows.append("Fixed effects & " + " & ".join(["firm + date"] * len(DEP_ORDER)) + " \\\\")
    rows.append("SE type & " + " & ".join(["firm clustered"] * len(DEP_ORDER)) + " \\\\")
    rows.append("\\bottomrule")
    rows.append("\\end{tabular}")
    rows.append("\\vspace{0.35em}")
    rows.append("\\noindent\\parbox{\\textwidth}{\\footnotesize")
    rows.append(
        "Notes: This table reports the second-stage IV spline coefficients for the global Russell 2000 design. "
        "Standard errors are reported underneath coefficients in parentheses. "
        "The dependent-variable panels correspond to Amihud illiquidity, volatility, and price informativeness. "
        "* p$<$0.10, ** p$<$0.05, *** p$<$0.01.\\\\"
    )
    rows.append(
        "The spline basis is a natural cubic spline in `ind_own` with firm and date fixed effects and firm-clustered standard errors."
    )
    rows.append("}")
    rows.append("\\end{threeparttable}")
    rows.append("\\end{table}")
    return "\n".join(rows) + "\n"


def main() -> None:
    df = load_data()
    OUT_FIRST_STAGE_TEX.write_text(build_first_stage_table(df), encoding="utf-8")
    OUT_IV_TEX.write_text(build_iv_table(df), encoding="utf-8")
    print(f"Saved: {OUT_FIRST_STAGE_TEX}")
    print(f"Saved: {OUT_IV_TEX}")


if __name__ == "__main__":
    main()
