#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import pandas as pd


INPUT_CSV = Path("analysis/nonlinearity/nonlinear_quadratic_results.csv")
OUTPUT_TEX = Path("analysis/nonlinearity/nonlinear_quadratic_results_pretty.tex")

DEP_ORDER = ["amihud_illiq", "volatility", "price_info"]
DEP_LABELS = {
    "amihud_illiq": "Amihud illiquidity",
    "volatility": "Volatility",
    "price_info": "Price informativeness",
}


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
        "beta_ind_own",
        "se_ind_own",
        "p_ind_own",
        "beta_ind_own_sq",
        "se_ind_own_sq",
        "p_ind_own_sq",
        "turning_point",
        "turning_point_se",
        "ind_own_min",
        "ind_own_q01",
        "ind_own_q05",
        "ind_own_q50",
        "ind_own_q95",
        "ind_own_q99",
        "ind_own_max",
        "nobs",
        "entities",
        "time_periods",
    ]
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["dep_var"] = pd.Categorical(df["dep_var"], categories=DEP_ORDER, ordered=True)
    return df.sort_values("dep_var").reset_index(drop=True)


def pick_row(df: pd.DataFrame, dep_var: str) -> pd.Series:
    rows = df[df["dep_var"] == dep_var]
    if rows.empty:
        raise KeyError(f"Missing row for dep_var={dep_var}")
    return rows.iloc[0]


def fmt_num(x: float) -> str:
    return f"{x:.4f}"


def fmt_yesno(v) -> str:
    if pd.isna(v):
        return ""
    if isinstance(v, str):
        return v
    return "Yes" if bool(v) else "No"


def build_table(df: pd.DataFrame) -> str:
    rows: list[str] = []
    rows.append("\\begin{table}[!htbp]\\centering")
    rows.append("\\caption{Quadratic Nonlinearity Results}")
    rows.append("\\label{tab:nonlinear_quadratic_results}")
    rows.append("\\begin{threeparttable}")
    rows.append("\\footnotesize")
    rows.append("\\setlength{\\tabcolsep}{3.5pt}")
    rows.append("\\renewcommand{\\arraystretch}{1.05}")
    rows.append("\\begin{tabular}{lccc}")
    rows.append("\\toprule")
    rows.append(" & " + " & ".join(DEP_LABELS[d] for d in DEP_ORDER) + " \\\\")
    rows.append("\\midrule")

    rows.append(
        "Linear term ($b_1$) & "
        + " & ".join(
            fmt_cell(float(pick_row(df, dep)["beta_ind_own"]), float(pick_row(df, dep)["se_ind_own"]), float(pick_row(df, dep)["p_ind_own"]))
            for dep in DEP_ORDER
        )
        + " \\\\"
    )
    rows.append(
        "Quadratic term ($b_2$) & "
        + " & ".join(
            fmt_cell(
                float(pick_row(df, dep)["beta_ind_own_sq"]),
                float(pick_row(df, dep)["se_ind_own_sq"]),
                float(pick_row(df, dep)["p_ind_own_sq"]),
            )
            for dep in DEP_ORDER
        )
        + " \\\\"
    )
    rows.append(
        "Turning point & "
        + " & ".join(
            f"\\shortstack{{{float(pick_row(df, dep)['turning_point']):.4f}\\\\({float(pick_row(df, dep)['turning_point_se']):.4f})}}"
            for dep in DEP_ORDER
        )
        + " \\\\"
    )
    rows.append("In support & " + " & ".join(fmt_yesno(pick_row(df, dep)["turning_in_q01_q99"]) for dep in DEP_ORDER) + " \\\\")

    rows.append("\\midrule")
    summary_rows = [
        ("Observations", "nobs"),
        ("Entities", "entities"),
        ("Time periods", "time_periods"),
        ("Support min", "ind_own_min"),
        ("Support q01", "ind_own_q01"),
        ("Support q05", "ind_own_q05"),
        ("Support q50", "ind_own_q50"),
        ("Support q95", "ind_own_q95"),
        ("Support q99", "ind_own_q99"),
        ("Support max", "ind_own_max"),
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
        "Notes: This table reports the reduced-form quadratic fixed-effects results for the three market-quality outcomes. "
        "Coefficients are reported with standard errors underneath in parentheses, and significance stars are based on p-values. "
        "The turning-point row reports the implied optimum of the fitted quadratic and its standard error. "
        "* p$<$0.10, ** p$<$0.05, *** p$<$0.01.\\\\"
    )
    rows.append(
        "The support rows summarize the empirical distribution of `ind_own` used to interpret whether the turning point lies inside the observed range."
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
