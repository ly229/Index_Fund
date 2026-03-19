#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import pandas as pd


DETAIL_CSV = Path("analysis/business_cycle/business_cycle/bus_size/size_bus_cycle/panel_cycle_size_detailed.csv")
BEST_CSV = Path("analysis/business_cycle/business_cycle/bus_size/size_bus_cycle/panel_cycle_size_best_results.csv")

DETAIL_TEX_DIR = Path("analysis/business_cycle/business_cycle/bus_size/size_bus_cycle")
BEST_TEX = Path("analysis/business_cycle/business_cycle/bus_size/size_bus_cycle/panel_cycle_size_best_results_pretty.tex")

DETAIL_TEX_MAP = {
    "amihud_illiq": DETAIL_TEX_DIR / "panel_cycle_size_detailed_amihud_illiq_pretty.tex",
    "volatility": DETAIL_TEX_DIR / "panel_cycle_size_detailed_volatility_pretty.tex",
    "price_info": DETAIL_TEX_DIR / "panel_cycle_size_detailed_price_info_pretty.tex",
}

DEP_ORDER = ["amihud_illiq", "volatility", "price_info"]
DEP_LABELS = {
    "amihud_illiq": "Amihud illiquidity",
    "volatility": "Volatility",
    "price_info": "Price informativeness",
}

CYCLE_ORDER = [
    "exp_1993Q1_2000Q4",
    "rec_2001Q1_2001Q4",
    "exp_2002Q1_2007Q3",
    "rec_2007Q4_2009Q2",
    "exp_2009Q3_2019Q3",
    "rec_2019Q4_2020Q2",
    "exp_2020Q3_2023Q4",
]
CYCLE_LABELS = {
    "exp_1993Q1_2000Q4": "Exp. 1993-2000",
    "rec_2001Q1_2001Q4": "Rec. 2001",
    "exp_2002Q1_2007Q3": "Exp. 2002-2007",
    "rec_2007Q4_2009Q2": "Rec. 2007-2009",
    "exp_2009Q3_2019Q3": "Exp. 2009-2019",
    "rec_2019Q4_2020Q2": "Rec. 2019-2020",
    "exp_2020Q3_2023Q4": "Exp. 2020-2023",
}


def fmt_stars(stars: str | float) -> str:
    if pd.isna(stars):
        return ""
    return str(stars)


def fmt_cell(coef: float, se: float, stars: str) -> str:
    return f"\\shortstack{{{coef:.4f}{stars}\\\\({se:.4f})}}"


def lookup_detail_cell(sub: pd.DataFrame, dec: int, cycle: str) -> str:
    cell = sub[(sub["size_decile"] == dec) & (sub["cycle"] == cycle)]
    if cell.empty:
        return "—"
    r = cell.iloc[0]
    if pd.isna(r["beta_ind_own"]) or pd.isna(r["se_ind_own"]):
        return "—"
    return fmt_cell(float(r["beta_ind_own"]), float(r["se_ind_own"]), fmt_stars(r["stars"]))


def build_detailed_table(df: pd.DataFrame, dep: str) -> str:
    sub = df[df["dep_var"] == dep].copy()
    rows: list[str] = []
    rows.append("\\begin{table}[!htbp]\\centering")
    rows.append(f"\\caption{{Business-Cycle by Size-Decile Regression Results: {DEP_LABELS[dep]}}}")
    rows.append(f"\\label{{tab:cycle_size_detailed_{dep}}}")
    rows.append("\\begin{threeparttable}")
    rows.append("\\footnotesize")
    rows.append("\\setlength{\\tabcolsep}{3.2pt}")
    rows.append("\\renewcommand{\\arraystretch}{1.04}")

    rows.append("\\begin{tabular}{@{}l" + "c" * len(CYCLE_ORDER) + "@{}}")
    rows.append("\\toprule")
    rows.append("Size decile & " + " & ".join(CYCLE_LABELS[c] for c in CYCLE_ORDER) + " \\\\")
    rows.append("\\midrule")
    for dec in range(1, 11):
        cells = [lookup_detail_cell(sub, dec, cycle) for cycle in CYCLE_ORDER]
        rows.append(f"{dec} & " + " & ".join(cells) + " \\\\")
    rows.append("\\bottomrule")
    rows.append("\\end{tabular}")

    rows.append("\\vspace{0.35em}")
    rows.append("\\noindent\\parbox{\\textwidth}{\\footnotesize")
    rows.append(
        "Notes: Each cell reports the coefficient on industry ownership from a two-way fixed-effects regression "
        "estimated within a business-cycle segment and size decile. Standard errors are reported underneath coefficients in parentheses. "
        "* p$<$0.10, ** p$<$0.05, *** p$<$0.01.\\\\"
    )
    rows.append(
        "The cycle labels are abbreviated as expansion (Exp.) and recession (Rec.). Cell-level sample sizes, entity counts, "
        "and time-period counts vary across regressions and are stored in the source CSV. A dash indicates that the source regression cell is not available."
    )
    rows.append("}")
    rows.append("\\end{threeparttable}")
    rows.append("\\end{table}")
    return "\n".join(rows) + "\n"


def build_best_table(df: pd.DataFrame) -> str:
    def panel_block(rank_type: str, title: str, is_dep_panel: bool = False) -> str:
        sub = df[df["rank_type"] == rank_type].copy()
        sub = sub.sort_values(["score", "dep_var", "cycle", "size_decile"], ascending=[False, True, True, True]).reset_index(drop=True)
        block: list[str] = []
        block.append("\\parbox[t]{\\linewidth}{")
        block.append("\\centering")
        block.append(f"\\textbf{{{title}}}")
        block.append("\\par\\vspace{0.25em}")
        block.append("\\scriptsize")
        block.append("\\setlength{\\tabcolsep}{2.1pt}")
        block.append("\\renewcommand{\\arraystretch}{0.95}")
        block.append("\\begin{tabular}{@{}rllrccc@{}}")
        block.append("\\toprule")
        block.append("Rank & Dep. var. & Cycle & Decile & $\\beta$ (SE) & $p$ & Score \\\\")
        block.append("\\midrule")
        if not is_dep_panel:
            for i, r in sub.iterrows():
                block.append(
                    f"{i + 1} & {DEP_LABELS[r['dep_var']]} & {CYCLE_LABELS[r['cycle']]} & {int(r['size_decile'])} & "
                    f"\\shortstack{{{float(r['beta_ind_own']):.4f}{fmt_stars(r['stars'])}\\\\({float(r['se_ind_own']):.4f})}} & "
                    f"{float(r['p_ind_own']):.4f} & {float(r['score']):.2f} \\\\"
                )
        else:
            rank = 1
            for dep in DEP_ORDER:
                dep_sub = sub[sub["dep_var"] == dep].copy().sort_values("score", ascending=False).reset_index(drop=True)
                block.append(f"\\multicolumn{{7}}{{l}}{{\\textit{{{DEP_LABELS[dep]}}}}} \\\\")
                for _, r in dep_sub.iterrows():
                    block.append(
                        f"{rank} & \\multicolumn{{1}}{{l}}{{}} & {CYCLE_LABELS[r['cycle']]} & {int(r['size_decile'])} & "
                        f"\\shortstack{{{float(r['beta_ind_own']):.4f}{fmt_stars(r['stars'])}\\\\({float(r['se_ind_own']):.4f})}} & "
                        f"{float(r['p_ind_own']):.4f} & {float(r['score']):.2f} \\\\"
                    )
                    rank += 1
        block.append("\\bottomrule")
        block.append("\\end{tabular}")
        block.append("}")
        return "\n".join(block)

    rows: list[str] = []
    rows.append("\\begin{table}[!htbp]\\centering")
    rows.append("\\caption{Best Business-Cycle by Size-Decile Results}")
    rows.append("\\label{tab:cycle_size_best_results}")
    rows.append("\\begin{threeparttable}")
    rows.append("\\footnotesize")
    rows.append("\\setlength{\\tabcolsep}{2.4pt}")
    rows.append("\\renewcommand{\\arraystretch}{0.98}")
    rows.append("\\noindent\\begin{tabular}{@{}p{0.495\\textwidth}p{0.495\\textwidth}@{}}")
    rows.append(
        panel_block("global_top15", "Panel A: Global top 15 significant cells")
        + " & "
        + panel_block("dep_top5", "Panel B: Top 5 significant cells by dependent variable", is_dep_panel=True)
    )
    rows.append("\\end{tabular}")
    rows.append("\\vspace{0.35em}")
    rows.append("\\noindent\\parbox{\\textwidth}{\\footnotesize")
    rows.append(
        "Notes: The table lists the highest-scoring significant coefficient estimates on industry ownership. "
        "Score is defined in the source workflow as |t| $\\times$ log(1 + N). Standard errors are reported underneath coefficients in parentheses. "
        "* p$<$0.10, ** p$<$0.05, *** p$<$0.01.\\\\"
    )
    rows.append(
        "The global panel ranks all significant cells, while the dependent-variable panel keeps the top five cells within each outcome group."
    )
    rows.append("}")
    rows.append("\\end{threeparttable}")
    rows.append("\\end{table}")
    return "\n".join(rows) + "\n"


def main() -> None:
    detail = pd.read_csv(DETAIL_CSV)
    best = pd.read_csv(BEST_CSV)

    for dep, out_path in DETAIL_TEX_MAP.items():
        out_path.write_text(build_detailed_table(detail, dep), encoding="utf-8")
    BEST_TEX.write_text(build_best_table(best), encoding="utf-8")

    for out_path in DETAIL_TEX_MAP.values():
        print(f"Saved: {out_path}")
    print(f"Saved: {BEST_TEX}")


if __name__ == "__main__":
    main()
