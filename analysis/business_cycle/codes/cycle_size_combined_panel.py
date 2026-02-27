#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DATA_PATH = Path("data/df_regression.csv")
OUT_DIR = Path("analysis/business_cycle")

DEP_VARS = ["amihud_illiq", "volatility", "price_info"]
RHS_VARS = ["ind_own", "c_firm_size", "c_dollar_vol"]

# Same cycle definitions used in existing cycle analysis.
CYCLES = [
    ("exp_1993Q1_2000Q4", "1993Q1", "2000Q4"),
    ("rec_2001Q1_2001Q4", "2001Q1", "2001Q4"),
    ("exp_2002Q1_2007Q3", "2002Q1", "2007Q3"),
    ("rec_2007Q4_2009Q2", "2007Q4", "2009Q2"),
    ("exp_2009Q3_2019Q3", "2009Q3", "2019Q3"),
    ("rec_2019Q4_2020Q2", "2019Q4", "2020Q2"),
    ("exp_2020Q3_2023Q4", "2020Q3", "2023Q4"),
]

OUT_DETAIL = OUT_DIR / "panel_cycle_size_detailed.csv"
OUT_BEST = OUT_DIR / "panel_cycle_size_best_results.csv"
OUT_SUMMARY_MD = OUT_DIR / "panel_cycle_size_best_summary.md"


def stars(p: float) -> str:
    if p < 0.01:
        return "***"
    if p < 0.05:
        return "**"
    if p < 0.10:
        return "*"
    return ""


def add_size_deciles(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["mktcap"] = pd.to_numeric(d["mktcap"], errors="coerce")
    d = d[d["mktcap"].notna()].copy()
    # Observation-level mktcap deciles to align with prior size-sorted outputs.
    d["size_decile"] = pd.qcut(d["mktcap"], 10, labels=False, duplicates="drop") + 1
    return d


def prep_cell(df: pd.DataFrame, dep_var: str, q_start: str, q_end: str, decile: int) -> pd.DataFrame:
    cols = ["cusip", "date", "size_decile", dep_var] + RHS_VARS
    d = df[cols].copy()
    d["date"] = pd.to_datetime(d["date"])
    d["quarter"] = d["date"].dt.to_period("Q").astype(str)
    d = d[(d["size_decile"] == decile) & (d["quarter"] >= q_start) & (d["quarter"] <= q_end)].copy()

    for c in [dep_var] + RHS_VARS:
        d[c] = pd.to_numeric(d[c], errors="coerce")
        d.loc[~np.isfinite(d[c]), c] = np.nan

    d = d.dropna(subset=[dep_var] + RHS_VARS)
    d = d.set_index(["cusip", "date"]).sort_index()
    return d


def run_cell(df: pd.DataFrame, dep_var: str, cycle_name: str, q_start: str, q_end: str, decile: int) -> dict:
    d = prep_cell(df, dep_var, q_start, q_end, decile)
    if d.empty:
        return {}
    if d.index.get_level_values(0).nunique() < 20:
        return {}
    if d.index.get_level_values(1).nunique() < 3:
        return {}

    formula = f"{dep_var} ~ 1 + {' + '.join(RHS_VARS)} + EntityEffects + TimeEffects"
    model = PanelOLS.from_formula(formula, data=d, drop_absorbed=True)
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            fit = model.fit(cov_type="clustered", cluster_entity=True, cluster_time=True)
            beta = float(fit.params["ind_own"])
            se = float(fit.std_errors["ind_own"])
            p = float(fit.pvalues["ind_own"])
            t = beta / se
    except Exception:
        return {}

    return {
        "dep_var": dep_var,
        "cycle": cycle_name,
        "size_decile": decile,
        "beta_ind_own": beta,
        "se_ind_own": se,
        "t_ind_own": t,
        "p_ind_own": p,
        "stars": stars(p),
        "ci95_lo": beta - 1.96 * se,
        "ci95_hi": beta + 1.96 * se,
        "nobs": int(fit.nobs),
        "entities": int(d.index.get_level_values(0).nunique()),
        "time_periods": int(d.index.get_level_values(1).nunique()),
    }


def build_best_table(detail: pd.DataFrame) -> pd.DataFrame:
    d = detail.copy()
    d["abs_t"] = d["t_ind_own"].abs()
    d["abs_beta"] = d["beta_ind_own"].abs()
    d["score"] = d["abs_t"] * np.log1p(d["nobs"])
    sig = d[d["p_ind_own"] < 0.05].copy()

    best_by_dep = (
        sig.sort_values(["dep_var", "score"], ascending=[True, False])
        .groupby("dep_var", as_index=False)
        .head(5)
        .copy()
    )
    best_global = sig.sort_values("score", ascending=False).head(15).copy()
    best_global["rank_type"] = "global_top15"
    best_by_dep["rank_type"] = "dep_top5"

    out = pd.concat([best_global, best_by_dep], ignore_index=True)
    out = out[
        [
            "rank_type",
            "dep_var",
            "cycle",
            "size_decile",
            "beta_ind_own",
            "se_ind_own",
            "t_ind_own",
            "p_ind_own",
            "stars",
            "ci95_lo",
            "ci95_hi",
            "nobs",
            "entities",
            "time_periods",
            "score",
        ]
    ]
    return out


def write_summary_md(best: pd.DataFrame) -> None:
    lines = []
    lines.append("# Combined Business-Cycle × Size Results")
    lines.append("")
    lines.append("Ranking rule: significant cells (p<0.05), sorted by |t| * log(1+N).")
    lines.append("")
    for dep in DEP_VARS:
        sub = best[(best["rank_type"] == "dep_top5") & (best["dep_var"] == dep)].head(1)
        if sub.empty:
            continue
        r = sub.iloc[0]
        lines.append(
            f"- Best `{dep}`: cycle={r['cycle']}, decile={int(r['size_decile'])}, "
            f"beta={r['beta_ind_own']:.4f}, t={r['t_ind_own']:.2f}, p={r['p_ind_own']:.4g}, N={int(r['nobs'])}."
        )
    lines.append("")
    OUT_SUMMARY_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_heatmaps(detail: pd.DataFrame) -> None:
    cycle_order = [c[0] for c in CYCLES]
    for dep in DEP_VARS:
        sub = detail[detail["dep_var"] == dep].copy()
        piv = sub.pivot_table(index="size_decile", columns="cycle", values="t_ind_own")
        piv = piv.reindex(index=range(1, 11), columns=cycle_order)
        mat = piv.to_numpy(dtype=float)

        vmax = np.nanpercentile(np.abs(mat), 95)
        vmax = max(vmax, 2.0)
        vmin = -vmax

        fig, ax = plt.subplots(figsize=(12.5, 4.8))
        im = ax.imshow(mat, aspect="auto", cmap="RdBu_r", vmin=vmin, vmax=vmax)
        ax.set_title(f"t-stat of Beta(ind_own): {dep} (cycle × size decile)")
        ax.set_xlabel("Business cycle segment")
        ax.set_ylabel("Size decile (1=small, 10=large)")
        ax.set_xticks(np.arange(len(cycle_order)))
        ax.set_xticklabels(cycle_order, rotation=35, ha="right", fontsize=8)
        ax.set_yticks(np.arange(10))
        ax.set_yticklabels([str(i) for i in range(1, 11)])
        cbar = fig.colorbar(im, ax=ax, shrink=0.9)
        cbar.set_label("t-stat on ind_own")
        fig.tight_layout()
        fig.savefig(OUT_DIR / f"heatmap_t_indown_cycle_size_{dep}.svg", format="svg", dpi=220, bbox_inches="tight")
        plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(DATA_PATH)
    df = add_size_deciles(raw)

    rows = []
    for cycle_name, q_start, q_end in CYCLES:
        for dec in range(1, 11):
            for dep in DEP_VARS:
                r = run_cell(df, dep, cycle_name, q_start, q_end, dec)
                if r:
                    rows.append(r)

    detail = pd.DataFrame(rows).sort_values(["dep_var", "cycle", "size_decile"]).reset_index(drop=True)
    detail.to_csv(OUT_DETAIL, index=False)

    best = build_best_table(detail)
    best.to_csv(OUT_BEST, index=False)
    write_summary_md(best)
    plot_heatmaps(detail)

    print(f"Saved: {OUT_DETAIL}")
    print(f"Saved: {OUT_BEST}")
    print(f"Saved: {OUT_SUMMARY_MD}")
    for dep in DEP_VARS:
        print(f"Saved: {OUT_DIR / f'heatmap_t_indown_cycle_size_{dep}.svg'}")


if __name__ == "__main__":
    main()
