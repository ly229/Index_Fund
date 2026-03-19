#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import pandas as pd


INPUT_CSV = Path("analysis/nonlinearity/iv_quadratic_cutoff1000_vs_2000_comparison.csv")
OUTPUT_TEX = Path("analysis/nonlinearity/iv_quadratic_cutoff1000_vs_2000_comparison_pretty.tex")

DEP_ORDER = ["amihud_illiq", "volatility", "price_info"]
DEP_LABELS = {
    "amihud_illiq": "Amihud illiquidity",
    "volatility": "Volatility",
    "price_info": "Price informativeness",
}

CUTOFF_ORDER = [1000, 2000]
CUTOFF_LABELS = {
    1000: "Russell 1000",
    2000: "Russell 2000",
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


def fmt_p(p: float) -> str:
    if pd.isna(p):
        return ""
    if p <= 0:
        return "<1e-16"
    if p < 0.0001:
        return f"{p:.2e}"
    return f"{p:.4f}"


def fmt_coef(beta: float, pval: float) -> str:
    return f"{beta:.4f}{stars_from_p(pval)}"


def fmt_fstat(fstat: float) -> str:
    return f"{fstat:.3f}"


def load_data() -> pd.DataFrame:
    df = pd.read_csv(INPUT_CSV)
    num_cols = [
        "cutoff",
        "beta_ind_own",
        "p_ind_own",
        "beta_ind_own_sq",
        "p_ind_own_sq",
        "turning_point",
        "first_stage_jointF_ind_own",
        "first_stage_jointF_ind_own_sq",
    ]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["turning_in_q01_q99"] = df["turning_in_q01_q99"].astype(str).str.lower().map({"true": "Yes", "false": "No"})
    df["cutoff"] = pd.Categorical(df["cutoff"], categories=CUTOFF_ORDER, ordered=True)
    df["dep_var"] = pd.Categorical(df["dep_var"], categories=DEP_ORDER, ordered=True)
    return df.sort_values(["cutoff", "dep_var"]).reset_index(drop=True)


def pick_row(df: pd.DataFrame, cutoff: int, dep_var: str) -> pd.Series:
    rows = df[(df["cutoff"] == cutoff) & (df["dep_var"] == dep_var)]
    if rows.empty:
        raise KeyError(f"Missing row for cutoff={cutoff}, dep_var={dep_var}")
    return rows.iloc[0]


def build_table_for_cutoff(df: pd.DataFrame, cutoff: int) -> str:
    rows: list[str] = []
    label = CUTOFF_LABELS[cutoff]
    suffix = "r1000" if cutoff == 1000 else "r2000"
    rows.append("\\begin{table}[!htbp]\\centering")
    rows.append(f"\\caption{{Quadratic IV Comparison: {label}}}")
    rows.append(f"\\label{{tab:iv_quadratic_comparison_{suffix}}}")
    rows.append("\\begin{threeparttable}")
    rows.append("\\footnotesize")
    rows.append("\\setlength{\\tabcolsep}{3.5pt}")
    rows.append("\\renewcommand{\\arraystretch}{1.05}")
    rows.append("\\begin{tabular}{lcccccc}")
    rows.append("\\toprule")
    rows.append(" & b1 & b2 & Turning point & In support & F(own) & F(own$^2$) \\\\")
    rows.append("\\midrule")

    for dep in DEP_ORDER:
        r = pick_row(df, cutoff, dep)
        rows.append(
            f"{DEP_LABELS[dep]} & "
            f"{fmt_coef(float(r['beta_ind_own']), float(r['p_ind_own']))} & "
            f"{fmt_coef(float(r['beta_ind_own_sq']), float(r['p_ind_own_sq']))} & "
            f"{float(r['turning_point']):.4f} & "
            f"{r['turning_in_q01_q99']} & "
            f"{fmt_fstat(float(r['first_stage_jointF_ind_own']))} & "
            f"{fmt_fstat(float(r['first_stage_jointF_ind_own_sq']))} \\\\"
        )

    rows.append("\\midrule")
    rows.append(
        "\\multicolumn{6}{l}{Controls: `c_firm_size`, `c_dollar_vol`, `running`, `running^2`. Firm and time fixed effects. Firm-clustered standard errors.} \\\\"
    )
    rows.append("\\multicolumn{6}{l}{Sample size is not reported in this comparison CSV.} \\\\")
    rows.append("\\bottomrule")
    rows.append("\\end{tabular}")
    rows.append("\\vspace{0.35em}")
    rows.append("\\noindent\\parbox{\\textwidth}{\\footnotesize")
    rows.append(
        "Notes: This table reports the quadratic IV comparison for the global design at the Russell "
        f"{cutoff} cutoff. Coefficient entries are reported with significance stars based on p-values. "
        "The turning-point indicator reports whether the estimated turning point lies inside the [q01, q99] support band. "
        "* p$<$0.10, ** p$<$0.05, *** p$<$0.01.\\\\"
    )
    rows.append("The first-stage columns report the joint F statistics for `ind_own` and `ind_own^2`.")
    rows.append("Sample size is not reported in this comparison CSV.")
    rows.append("}")
    rows.append("\\end{threeparttable}")
    rows.append("\\end{table}")
    return "\n".join(rows) + "\n"


def main() -> None:
    df = load_data()
    tex = build_table_for_cutoff(df, 1000) + "\n" + build_table_for_cutoff(df, 2000)
    OUTPUT_TEX.write_text(tex, encoding="utf-8")
    print(f"Saved: {OUTPUT_TEX}")


if __name__ == "__main__":
    main()
