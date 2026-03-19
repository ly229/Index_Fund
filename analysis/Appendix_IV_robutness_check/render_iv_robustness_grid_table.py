#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import pandas as pd


INPUT_CSV = Path("analysis/Appendix_IV_robutness_check/iv_robustness_grid_results.csv")
OUTPUT_TEX = Path("analysis/Appendix_IV_robutness_check/iv_robustness_grid_results_pretty.tex")

DEP_ORDER = ["amihud_illiq", "volatility", "price_info"]
DEP_LABELS = {
    "amihud_illiq": "Amihud illiquidity",
    "volatility": "Volatility",
    "price_info": "Price informativeness",
}

BANDWIDTH_ORDER = [50.0, 100.0, 150.0, 200.0, 300.0, 500.0, None]

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


def bw_label(bw: float | None) -> str:
    return "global" if bw is None else str(int(bw))


def fmt_cell(beta: float, se: float, pval: float, fstat: float, fp: float) -> str:
    return f"\\shortstack{{{beta:.4f}{stars_from_p(pval)}\\\\({se:.4f})\\\\F={fstat:.2f}\\\\p={fmt_p(fp)}}}"


def load_data() -> pd.DataFrame:
    df = pd.read_csv(INPUT_CSV)
    for col in ["cutoff", "bandwidth", "nobs", "n_firms", "n_dates", "n_clusters", "beta_endog", "se_endog", "p_endog", "first_stage_f", "first_stage_p"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["dep_var"] = pd.Categorical(df["dep_var"], categories=DEP_ORDER, ordered=True)
    df["cutoff"] = pd.Categorical(df["cutoff"], categories=CUTOFF_ORDER, ordered=True)
    # Preserve bandwidth ordering manually, with global last.
    df["bw_sort"] = df["bandwidth"].map(lambda x: 999999 if pd.isna(x) else float(x))
    return df.sort_values(["cutoff", "dep_var", "bw_sort"]).reset_index(drop=True)


def pick_row(df: pd.DataFrame, cutoff: float, dep_var: str, bw: float | None) -> pd.Series:
    if bw is None:
        rows = df[(df["cutoff"] == cutoff) & (df["dep_var"] == dep_var) & (df["bandwidth"].isna())]
    else:
        rows = df[(df["cutoff"] == cutoff) & (df["dep_var"] == dep_var) & (df["bandwidth"] == bw)]
    if rows.empty:
        raise KeyError(f"Missing row for cutoff={cutoff}, dep_var={dep_var}, bandwidth={bw}")
    return rows.iloc[0]


def build_table_for_cutoff(df: pd.DataFrame, cutoff: float) -> str:
    label = CUTOFF_LABELS[cutoff]
    suffix = "r1000" if cutoff == 1000.0 else "r2000"

    rows: list[str] = []
    rows.append("\\begin{table}[!htbp]\\centering")
    rows.append(f"\\caption{{IV Robustness Grid: {label}}}")
    rows.append(f"\\label{{tab:iv_robustness_grid_{suffix}}}")
    rows.append("\\begin{threeparttable}")
    rows.append("\\footnotesize")
    rows.append("\\setlength{\\tabcolsep}{2.8pt}")
    rows.append("\\renewcommand{\\arraystretch}{1.05}")
    rows.append("\\begin{tabular}{l" + "c" * len(BANDWIDTH_ORDER) + "}")
    rows.append("\\toprule")
    rows.append(" & " + " & ".join(bw_label(bw) for bw in BANDWIDTH_ORDER) + " \\\\")
    rows.append("\\midrule")

    for dep in DEP_ORDER:
        cells = []
        for bw in BANDWIDTH_ORDER:
            r = pick_row(df, cutoff, dep, bw)
            cells.append(
                fmt_cell(
                    float(r["beta_endog"]),
                    float(r["se_endog"]),
                    float(r["p_endog"]),
                    float(r["first_stage_f"]),
                    float(r["first_stage_p"]),
                )
            )
        rows.append(f"{DEP_LABELS[dep]} & " + " & ".join(cells) + " \\\\")

    rows.append("\\midrule")
    rows.append(
        "N & "
        + " & ".join(str(int(pick_row(df, cutoff, DEP_ORDER[0], bw)["nobs"])) for bw in BANDWIDTH_ORDER)
        + " \\\\"
    )
    rows.append(
        "Firms & "
        + " & ".join(str(int(pick_row(df, cutoff, DEP_ORDER[0], bw)["n_firms"])) for bw in BANDWIDTH_ORDER)
        + " \\\\"
    )
    rows.append(
        "Dates & "
        + " & ".join(str(int(pick_row(df, cutoff, DEP_ORDER[0], bw)["n_dates"])) for bw in BANDWIDTH_ORDER)
        + " \\\\"
    )
    rows.append(
        "Clusters & "
        + " & ".join(str(int(pick_row(df, cutoff, DEP_ORDER[0], bw)["n_clusters"])) for bw in BANDWIDTH_ORDER)
        + " \\\\"
    )
    rows.append("Controls & " + " & ".join(["`c_firm_size`, `c_dollar_vol`, `running` powers 1-2"] * len(BANDWIDTH_ORDER)) + " \\\\")
    rows.append("Fixed effects & " + " & ".join(["Firm + time"] * len(BANDWIDTH_ORDER)) + " \\\\")
    rows.append("SE type & " + " & ".join(["Firm clustered"] * len(BANDWIDTH_ORDER)) + " \\\\")
    rows.append("\\bottomrule")
    rows.append("\\end{tabular}")
    rows.append("\\vspace{0.35em}")
    rows.append("\\noindent\\parbox{\\textwidth}{\\footnotesize")
    rows.append(
        "Notes: This table reports the quadratic IV robustness grid for the local-window specifications at the "
        f"{label} cutoff. Each cell reports the 2SLS coefficient on `ind_own`, its standard error, and the first-stage F statistic / p-value. "
        "Bandwidth labels are in basis points; `global` denotes the unrestricted sample. "
        "* p$<$0.10, ** p$<$0.05, *** p$<$0.01.\\\\"
    )
    rows.append(
        "Weak local-window cases should be interpreted cautiously, especially when the first-stage F statistic is far below the conventional threshold of 10."
    )
    rows.append("}")
    rows.append("\\end{threeparttable}")
    rows.append("\\end{table}")
    return "\n".join(rows) + "\n"


def main() -> None:
    df = load_data()
    tex = build_table_for_cutoff(df, 1000.0) + "\n" + build_table_for_cutoff(df, 2000.0)
    OUTPUT_TEX.write_text(tex, encoding="utf-8")
    print(f"Saved: {OUTPUT_TEX}")


if __name__ == "__main__":
    main()
