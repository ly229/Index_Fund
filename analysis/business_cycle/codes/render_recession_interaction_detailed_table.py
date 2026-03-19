#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import pandas as pd


INPUT_CSV = Path("analysis/business_cycle/business_cycle/reg_beta/reg/panel_recession_interaction_detailed.csv")
OUTPUT_TEX = Path("analysis/business_cycle/business_cycle/reg_beta/reg/panel_recession_interaction_detailed_pretty.tex")

DEP_ORDER = ["amihud_illiq", "volatility", "price_info"]
DEP_LABELS = {
    "amihud_illiq": "Amihud illiquidity",
    "volatility": "Volatility",
    "price_info": "Price informativeness",
}

RHS_ORDER = ["ind_own", "ind_own_x_recession", "c_firm_size", "c_dollar_vol"]
RHS_LABELS = {
    "ind_own": "Industry ownership",
    "ind_own_x_recession": "Industry ownership $\\times$ recession",
    "c_firm_size": "Firm size",
    "c_dollar_vol": "Dollar volume",
}


def fmt_cell(coef: float, se: float, stars: str) -> str:
    return f"\\shortstack{{{coef:.4f}{stars}\\\\({se:.4f})}}"


def build_table(df: pd.DataFrame) -> str:
    piv = df.set_index(["rhs_var", "dep_var"]).sort_index()

    rows: list[str] = []
    rows.append("\\begin{table}[!htbp]\\centering")
    rows.append("\\caption{Heterogeneity by Recession: Panel FE with Interaction}")
    rows.append("\\label{tab:recession_interaction_detailed}")
    rows.append("\\begin{threeparttable}")
    rows.append("\\footnotesize")
    rows.append("\\setlength{\\tabcolsep}{3.5pt}")
    rows.append("\\renewcommand{\\arraystretch}{1.05}")
    rows.append("\\begin{tabular}{l" + "c" * len(DEP_ORDER) + "}")
    rows.append("\\toprule")
    rows.append(" & " + " & ".join(DEP_LABELS[d] for d in DEP_ORDER) + " \\\\")
    rows.append("\\midrule")

    for rhs in RHS_ORDER:
        row_cells = []
        for dep in DEP_ORDER:
            r = piv.loc[(rhs, dep)]
            stars = str(r.get("stars", "")) if pd.notna(r.get("stars", "")) else ""
            row_cells.append(fmt_cell(float(r["coef"]), float(r["se"]), stars))
        rows.append(f"{RHS_LABELS[rhs]} & " + " & ".join(row_cells) + " \\\\")

    rows.append("\\midrule")
    stats = [
        ("Observations", "nobs"),
        ("Entities", "entities"),
        ("Time periods", "time_periods"),
    ]
    for label, col in stats:
        rows.append(
            label
            + " & "
            + " & ".join(str(int(piv.loc[("ind_own", dep)][col])) for dep in DEP_ORDER)
            + " \\\\"
        )
    rows.append("Firm FE & Yes & Yes & Yes \\\\")
    rows.append("Time FE & Yes & Yes & Yes \\\\")
    rows.append("Two-way clustered SE & Yes & Yes & Yes \\\\")
    rows.append("\\bottomrule")
    rows.append("\\end{tabular}")
    rows.append("\\vspace{0.35em}")
    rows.append("\\noindent\\parbox{\\textwidth}{\\footnotesize")
    rows.append("Notes: This table reports panel fixed-effects regressions with an interaction between industry ownership and recession periods. Standard errors are reported underneath coefficients in parentheses. * p$<$0.10, ** p$<$0.05, *** p$<$0.01.\\\\")
    rows.append("The interaction coefficient is the incremental slope in recession quarters relative to non-recession quarters. Standard errors are two-way clustered by firm and date.")
    rows.append("}")
    rows.append("\\end{threeparttable}")
    rows.append("\\end{table}")
    return "\n".join(rows) + "\n"


def main() -> None:
    df = pd.read_csv(INPUT_CSV)
    OUTPUT_TEX.write_text(build_table(df), encoding="utf-8")
    print(f"Saved: {OUTPUT_TEX}")


if __name__ == "__main__":
    main()
