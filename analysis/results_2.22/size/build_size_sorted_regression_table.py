#!/usr/bin/env python3
from __future__ import annotations

from math import erfc, sqrt
from pathlib import Path

import pandas as pd


INPUT_CSV = Path("analysis/results_2.22/size/size_sorted_regression_results.csv")
OUTPUT_TEX = Path("analysis/results_2.22/size/size_sorted_regression_results_pretty.tex")

DEP_LABELS = {
    "amihud_illiq": "Amihud illiquidity",
    "volatility": "Volatility",
    "price_info": "Price informativeness",
}

VAR_ORDER = ["beta_ind_own", "beta_c_firm_size", "beta_c_dollar_vol"]
VAR_LABELS = {
    "beta_ind_own": "Industry ownership",
    "beta_c_firm_size": "Firm size",
    "beta_c_dollar_vol": "Dollar volume",
}
PANEL_ORDER = ["amihud_illiq", "volatility", "price_info"]


def stars_from_t(t_stat: float) -> str:
    pval = erfc(abs(t_stat) / sqrt(2.0))
    if pval < 0.01:
        return "***"
    if pval < 0.05:
        return "**"
    if pval < 0.10:
        return "*"
    return ""


def fmt_coef(beta: float, t_stat: float) -> str:
    return f"{beta:.4f}{stars_from_t(t_stat)}"


def fmt_cell(beta: float, se: float, t_stat: float) -> str:
    return f"\\shortstack{{{fmt_coef(beta, t_stat)}\\\\({se:.4f})}}"


def build_one_table(df: pd.DataFrame, dep: str) -> str:
    subdf = df[df["dep"] == dep].copy()
    deciles = sorted(int(d) for d in subdf["decile"].dropna().unique())
    rows: list[str] = []
    rows.append("\\begin{table}[!htbp]\\centering")
    rows.append(f"\\caption{{Size-Sorted Regression Results: {DEP_LABELS[dep]}}}")
    rows.append(f"\\label{{tab:size_sorted_regression_results_{dep}}}")
    rows.append("\\begin{threeparttable}")
    rows.append("\\begin{tabular}{l" + "c" * len(deciles) + "}")
    rows.append("\\toprule")
    rows.append(" & " + " & ".join(str(d) for d in deciles) + " \\\\")
    rows.append("\\midrule")

    for var, label in VAR_LABELS.items():
        cells = []
        for dec in deciles:
            r = subdf[subdf["decile"] == dec].iloc[0]
            beta = float(r[var])
            se = float(r[var.replace("beta", "se")])
            t_stat = float(r[var.replace("beta", "t")])
            cells.append(fmt_cell(beta, se, t_stat))
        rows.append(f"{label} & " + " & ".join(cells) + " \\\\")
    rows.append("\\midrule")
    rows.append("Observations & " + " & ".join(str(int(subdf[subdf['decile'] == dec].iloc[0]["N"])) for dec in deciles) + " \\\\")
    rows.append("within $R^2$ & " + " & ".join(f"{float(subdf[subdf['decile'] == dec].iloc[0]['within_R2']):.4f}" for dec in deciles) + " \\\\")

    rows.append("\\bottomrule")
    rows.append("\\end{tabular}")
    rows.append("\\begin{tablenotes}\\footnotesize")
    rows.append(
        "\\item Notes: The table reports observation-level two-way fixed-effects regressions within market-cap deciles. "
        "Standard errors are reported underneath coefficients in parentheses. "
        "* p$<$0.10, ** p$<$0.05, *** p$<$0.01."
    )
    rows.append(
        "\\item Sample construction: deciles are based on market capitalization. Regressions are estimated separately within each decile for each outcome variable."
    )
    rows.append("\\end{tablenotes}")
    rows.append("\\end{threeparttable}")
    rows.append("\\end{table}")
    return "\n".join(rows) + "\n"


def build_table(df: pd.DataFrame) -> str:
    rows: list[str] = []
    rows.append("\\begin{table}[!htbp]\\centering")
    rows.append("\\caption{Size-Sorted Regression Results by Market Capitalization Decile}")
    rows.append("\\label{tab:size_sorted_regression_results}")
    rows.append("\\begin{threeparttable}")
    rows.append("\\footnotesize")
    rows.append("\\setlength{\\tabcolsep}{1.5pt}")
    rows.append("\\renewcommand{\\arraystretch}{0.86}")

    def panel_block(dep: str, caption: str, width: str) -> list[str]:
        subdf = df[df["dep"] == dep].copy()
        deciles = sorted(int(d) for d in subdf["decile"].dropna().unique())
        block: list[str] = []
        block.append(f"\\begin{{subtable}}[t]{{{width}}}")
        block.append(f"\\caption{{{caption}}}")
        block.append("\\centering")
        block.append("\\begin{tabular}{@{}lccc@{}}")
        block.append("\\toprule")
        block.append("Decile & Ind. own. & Firm size & Dol. vol. \\\\")
        block.append("\\midrule")
        for dec in deciles:
            r = subdf[subdf["decile"] == dec].iloc[0]
            block.append(
                f"{dec} & "
                + " & ".join(
                    [
                        fmt_cell(float(r["beta_ind_own"]), float(r["se_ind_own"]), float(r["t_ind_own"])),
                        fmt_cell(float(r["beta_c_firm_size"]), float(r["se_c_firm_size"]), float(r["t_c_firm_size"])),
                        fmt_cell(float(r["beta_c_dollar_vol"]), float(r["se_c_dollar_vol"]), float(r["t_c_dollar_vol"])),
                    ]
                )
                + " \\\\"
            )
        block.append("\\midrule")
        block.append("Observations & \\multicolumn{3}{c}{" + str(int(subdf.iloc[0]["N"])) + "} \\\\")
        block.append("within $R^2$ & \\multicolumn{3}{c}{" + f"{float(subdf.iloc[0]['within_R2']):.4f}" + "} \\\\")
        block.append("\\bottomrule")
        block.append("\\end{tabular}")
        block.append("\\end{subtable}")
        return block

    rows.append("\\noindent\\begin{minipage}[t]{0.46\\textwidth}")
    rows.extend(panel_block("amihud_illiq", "Panel A: Amihud illiquidity", "\\linewidth"))
    rows.append("\\end{minipage}\\hfill")
    rows.append("\\begin{minipage}[t]{0.46\\textwidth}")
    rows.extend(panel_block("volatility", "Panel B: Volatility", "\\linewidth"))
    rows.append("\\end{minipage}")
    rows.append("\\vspace{0.5em}")
    rows.append("\\noindent\\begin{minipage}[t]{0.56\\textwidth}\\centering")
    rows.extend(panel_block("price_info", "Panel C: Price informativeness", "\\linewidth"))
    rows.append("\\end{minipage}")

    rows.append("\\vspace{0.35em}")
    rows.append("\\noindent\\parbox{\\textwidth}{\\footnotesize")
    rows.append(
        "Notes: This table reports the size-sorted regression results in three panels. "
        "Standard errors are reported underneath coefficients in parentheses. "
        "* p$<$0.10, ** p$<$0.05, *** p$<$0.01.\\\\"
    )
    rows.append(
        "Sample construction: deciles are based on market capitalization. Regressions are estimated separately within each decile for each outcome variable."
    )
    rows.append("}")
    rows.append("\\end{threeparttable}")
    rows.append("\\end{table}")
    return "\n".join(rows) + "\n"


def main() -> None:
    df = pd.read_csv(INPUT_CSV)
    pretty = build_table(df)
    OUTPUT_TEX.write_text(pretty, encoding="utf-8")
    print(f"Saved: {OUTPUT_TEX}")


if __name__ == "__main__":
    main()
