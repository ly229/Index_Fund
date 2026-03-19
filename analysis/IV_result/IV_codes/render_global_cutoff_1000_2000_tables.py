#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import pandas as pd


INPUT_CSV = Path("analysis/IV_result/iv_results_global_cutoff_1000_2000.csv")
OUT_FIRST_STAGE_TEX = Path("analysis/IV_result/iv_results_global_cutoff_1000_2000_first_stage_pretty.tex")
OUT_IV_TEX = Path("analysis/IV_result/iv_results_global_cutoff_1000_2000_iv_pretty.tex")

DEP_ORDER = ["amihud_illiq", "volatility", "price_info"]
DEP_LABELS = {
    "amihud_illiq": "Amihud illiquidity",
    "volatility": "Volatility",
    "price_info": "Price informativeness",
}

CUTOFF_ORDER = [1000.0, 2000.0]
CUTOFF_LABELS = {
    1000.0: "Russell 1000",
    2000.0: "Russell 2000",
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


def fmt_first_stage_cell(f_stat: float, pval: float) -> str:
    return f"\\shortstack{{{f_stat:.4f}\\\\(p={fmt_p(pval)})}}"


def fmt_iv_cell(beta: float, se: float, pval: float) -> str:
    return f"\\shortstack{{{beta:.4f}{stars_from_p(pval)}\\\\({se:.4f})}}"


def load_data() -> pd.DataFrame:
    df = pd.read_csv(INPUT_CSV)
    df["cutoff"] = pd.to_numeric(df["cutoff"], errors="coerce")
    df["nobs"] = pd.to_numeric(df["nobs"], errors="coerce")
    df["n_firms"] = pd.to_numeric(df["n_firms"], errors="coerce")
    df["n_dates"] = pd.to_numeric(df["n_dates"], errors="coerce")
    df["n_clusters"] = pd.to_numeric(df["n_clusters"], errors="coerce")
    df["beta_endog"] = pd.to_numeric(df["beta_endog"], errors="coerce")
    df["se_endog"] = pd.to_numeric(df["se_endog"], errors="coerce")
    df["t_endog"] = pd.to_numeric(df["t_endog"], errors="coerce")
    df["p_endog"] = pd.to_numeric(df["p_endog"], errors="coerce")
    df["first_stage_f"] = pd.to_numeric(df["first_stage_f"], errors="coerce")
    df["first_stage_p"] = pd.to_numeric(df["first_stage_p"], errors="coerce")
    df["dep_var"] = pd.Categorical(df["dep_var"], categories=DEP_ORDER, ordered=True)
    df["cutoff"] = pd.Categorical(df["cutoff"], categories=CUTOFF_ORDER, ordered=True)
    return df.sort_values(["dep_var", "cutoff"]).reset_index(drop=True)


def pick_row(df: pd.DataFrame, dep_var: str, cutoff: float) -> pd.Series:
    rows = df[(df["dep_var"] == dep_var) & (df["cutoff"] == cutoff)]
    if rows.empty:
        raise KeyError(f"Missing row for dep_var={dep_var}, cutoff={cutoff}")
    return rows.iloc[0]


def build_first_stage_table(df: pd.DataFrame) -> str:
    rows: list[str] = []
    rows.append("\\begin{table}[!htbp]\\centering")
    rows.append("\\caption{First-Stage F Statistics for Global IV Cutoffs}")
    rows.append("\\label{tab:iv_global_first_stage}")
    rows.append("\\begin{threeparttable}")
    rows.append("\\footnotesize")
    rows.append("\\setlength{\\tabcolsep}{3.5pt}")
    rows.append("\\renewcommand{\\arraystretch}{1.05}")
    rows.append("\\begin{tabular}{lcc}")
    rows.append("\\toprule")
    rows.append(" & " + " & ".join(CUTOFF_LABELS[c] for c in CUTOFF_ORDER) + " \\\\")
    rows.append("\\midrule")

    for dep in DEP_ORDER:
        cells = []
        for cutoff in CUTOFF_ORDER:
            r = pick_row(df, dep, cutoff)
            cells.append(fmt_first_stage_cell(float(r["first_stage_f"]), float(r["first_stage_p"])))
        rows.append(f"{DEP_LABELS[dep]} & " + " & ".join(cells) + " \\\\")

    rows.append("\\midrule")
    ref = pick_row(df, DEP_ORDER[0], CUTOFF_ORDER[0])
    rows.append("Observations & " + " & ".join(str(int(pick_row(df, dep, cutoff)["nobs"])) for cutoff in CUTOFF_ORDER for dep in [DEP_ORDER[0]]) + " \\\\")
    rows[-1] = "Observations & " + " & ".join(str(int(pick_row(df, DEP_ORDER[0], cutoff)["nobs"])) for cutoff in CUTOFF_ORDER) + " \\\\"
    rows.append("Firms & " + " & ".join(str(int(pick_row(df, DEP_ORDER[0], cutoff)["n_firms"])) for cutoff in CUTOFF_ORDER) + " \\\\")
    rows.append("Dates & " + " & ".join(str(int(pick_row(df, DEP_ORDER[0], cutoff)["n_dates"])) for cutoff in CUTOFF_ORDER) + " \\\\")
    rows.append("Clusters & " + " & ".join(str(int(pick_row(df, DEP_ORDER[0], cutoff)["n_clusters"])) for cutoff in CUTOFF_ORDER) + " \\\\")
    rows.append("Controls & " + " & ".join(["mktcap; c_dollar_vol; r_tno; rank_minus_cutoff"] * len(CUTOFF_ORDER)) + " \\\\")
    rows.append("Fixed effects & " + " & ".join(["cusip FE + date FE"] * len(CUTOFF_ORDER)) + " \\\\")
    rows.append("SE type & " + " & ".join(["firm clustered"] * len(CUTOFF_ORDER)) + " \\\\")
    rows.append("\\bottomrule")
    rows.append("\\end{tabular}")
    rows.append("\\vspace{0.35em}")
    rows.append("\\noindent\\parbox{\\textwidth}{\\footnotesize")
    rows.append(
        "Notes: This table reports first-stage relevance diagnostics for the global cutoff-IV design. "
        "Entries are first-stage F statistics for the excluded cutoff indicator in the first-stage regression of instrumented ownership. "
        "P-values are shown in parentheses. * p$<$0.10, ** p$<$0.05, *** p$<$0.01.\\\\"
    )
    rows.append("The sample is the global specification with no bandwidth restriction, estimated separately at the Russell 1000 and Russell 2000 cutoffs.")
    rows.append("}")
    rows.append("\\end{threeparttable}")
    rows.append("\\end{table}")
    return "\n".join(rows) + "\n"


def build_iv_table_for_cutoff(df: pd.DataFrame, cutoff: float) -> str:
    r = CUTOFF_LABELS[cutoff]
    rows: list[str] = []
    rows.append("\\begin{table}[!htbp]\\centering")
    rows.append(f"\\caption{{IV Regression Results: {r}}}")
    label_suffix = "r1000" if cutoff == 1000.0 else "r2000"
    rows.append(f"\\label{{tab:iv_global_results_{label_suffix}}}")
    rows.append("\\begin{threeparttable}")
    rows.append("\\footnotesize")
    rows.append("\\setlength{\\tabcolsep}{3.5pt}")
    rows.append("\\renewcommand{\\arraystretch}{1.05}")
    rows.append("\\begin{tabular}{lc}")
    rows.append("\\toprule")
    rows.append(" & " + r + " \\\\")
    rows.append("\\midrule")

    for dep in DEP_ORDER:
        row = pick_row(df, dep, cutoff)
        cell = fmt_iv_cell(float(row["beta_endog"]), float(row["se_endog"]), float(row["p_endog"]))
        rows.append(f"{DEP_LABELS[dep]} & {cell} \\\\")

    rows.append("\\midrule")
    summary_rows = [
        ("Observations", "nobs"),
        ("Firms", "n_firms"),
        ("Dates", "n_dates"),
        ("Clusters", "n_clusters"),
    ]
    for label, col in summary_rows:
        rows.append(label + " & " + str(int(pick_row(df, DEP_ORDER[0], cutoff)[col])) + " \\\\")
    rows.append("Endogenous variable & ind_own \\\\")
    rows.append("Instrument & z_rank_le_cutoff \\\\")
    rows.append("Controls & mktcap; c_dollar_vol; r_tno; rank_minus_cutoff \\\\")
    rows.append("Fixed effects & cusip FE + date FE \\\\")
    rows.append("SE type & firm clustered \\\\")
    rows.append("\\bottomrule")
    rows.append("\\end{tabular}")
    rows.append("\\vspace{0.35em}")
    rows.append("\\noindent\\parbox{\\textwidth}{\\footnotesize")
    rows.append(
        "Notes: This table reports second-stage IV estimates for the global cutoff-IV design. "
        "Standard errors are reported underneath coefficients in parentheses. "
        "* p$<$0.10, ** p$<$0.05, *** p$<$0.01.\\\\"
    )
    rows.append(f"The sample is the global specification with no bandwidth restriction, estimated at the {r} cutoff.")
    rows.append("}")
    rows.append("\\end{threeparttable}")
    rows.append("\\end{table}")
    return "\n".join(rows) + "\n"


def main() -> None:
    df = load_data()
    OUT_FIRST_STAGE_TEX.write_text(build_first_stage_table(df), encoding="utf-8")
    iv_text = build_iv_table_for_cutoff(df, 1000.0) + "\n" + build_iv_table_for_cutoff(df, 2000.0)
    OUT_IV_TEX.write_text(iv_text, encoding="utf-8")
    print(f"Saved: {OUT_FIRST_STAGE_TEX}")
    print(f"Saved: {OUT_IV_TEX}")


if __name__ == "__main__":
    main()
