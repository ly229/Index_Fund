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
OUT_DIR = Path("analysis/nonlinearity")

DEP_VARS = ["amihud_illiq", "volatility", "price_info"]
RHS_BASE = ["c_firm_size", "c_dollar_vol"]
MAIN = "ind_own"
MAIN_SQ = "ind_own_sq"

OUT_COEF = OUT_DIR / "nonlinear_quadratic_results.csv"
OUT_SCORE = OUT_DIR / "nonlinear_indexing_score_grid.csv"
OUT_SUMMARY = OUT_DIR / "nonlinear_indexing_summary.md"
OUT_PLOT_SCORE = OUT_DIR / "indexing_good_range_score.svg"


def prep(df: pd.DataFrame, dep_var: str) -> pd.DataFrame:
    cols = ["cusip", "date", MAIN, dep_var] + RHS_BASE
    d = df[cols].copy()
    d[MAIN_SQ] = d[MAIN] ** 2
    d["date"] = pd.to_datetime(d["date"])
    for c in [dep_var, MAIN, MAIN_SQ] + RHS_BASE:
        d[c] = pd.to_numeric(d[c], errors="coerce")
        d.loc[~np.isfinite(d[c]), c] = np.nan
    d = d.dropna(subset=[dep_var, MAIN, MAIN_SQ] + RHS_BASE)
    d = d.set_index(["cusip", "date"]).sort_index()
    return d


def fit_one(df: pd.DataFrame, dep_var: str) -> dict:
    d = prep(df, dep_var)
    formula = f"{dep_var} ~ 1 + {MAIN} + {MAIN_SQ} + {' + '.join(RHS_BASE)} + EntityEffects + TimeEffects"
    model = PanelOLS.from_formula(formula, data=d, drop_absorbed=True)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        fit = model.fit(cov_type="clustered", cluster_entity=True, cluster_time=True)
        b1 = float(fit.params[MAIN])
        b2 = float(fit.params[MAIN_SQ])
        se1 = float(fit.std_errors[MAIN])
        se2 = float(fit.std_errors[MAIN_SQ])
        p1 = float(fit.pvalues[MAIN])
        p2 = float(fit.pvalues[MAIN_SQ])
        cov = fit.cov.loc[[MAIN, MAIN_SQ], [MAIN, MAIN_SQ]].to_numpy(dtype=float)

    x = d[MAIN]
    q01, q05, q50, q95, q99 = [float(x.quantile(q)) for q in [0.01, 0.05, 0.50, 0.95, 0.99]]
    xmin, xmax = float(x.min()), float(x.max())

    turning = np.nan
    turn_se = np.nan
    in_support = False
    if abs(b2) > 1e-12:
        turning = -b1 / (2.0 * b2)
        # Delta-method SE for x* = -b1/(2*b2)
        d_db1 = -1.0 / (2.0 * b2)
        d_db2 = b1 / (2.0 * (b2 ** 2))
        grad = np.array([d_db1, d_db2], dtype=float)
        var_turn = float(grad.T @ cov @ grad)
        if var_turn > 0 and np.isfinite(var_turn):
            turn_se = np.sqrt(var_turn)
        in_support = (turning >= q01) and (turning <= q99)

    return {
        "dep_var": dep_var,
        "beta_ind_own": b1,
        "se_ind_own": se1,
        "p_ind_own": p1,
        "beta_ind_own_sq": b2,
        "se_ind_own_sq": se2,
        "p_ind_own_sq": p2,
        "turning_point": turning,
        "turning_point_se": turn_se,
        "turning_in_q01_q99": in_support,
        "ind_own_min": xmin,
        "ind_own_q01": q01,
        "ind_own_q05": q05,
        "ind_own_q50": q50,
        "ind_own_q95": q95,
        "ind_own_q99": q99,
        "ind_own_max": xmax,
        "nobs": int(fit.nobs),
        "entities": int(d.index.get_level_values(0).nunique()),
        "time_periods": int(d.index.get_level_values(1).nunique()),
    }


def partial_effect_curve(b1: float, b2: float, xgrid: np.ndarray, x_ref: float) -> np.ndarray:
    g = b1 * xgrid + b2 * (xgrid ** 2)
    g_ref = b1 * x_ref + b2 * (x_ref ** 2)
    return g - g_ref


def build_score_and_plot(coef_df: pd.DataFrame) -> pd.DataFrame:
    x_lo = float(coef_df["ind_own_q01"].max())
    x_hi = float(coef_df["ind_own_q99"].min())
    x_ref = float(coef_df["ind_own_q50"].median())
    xgrid = np.linspace(x_lo, x_hi, 300)

    curves = {}
    for dep in DEP_VARS:
        r = coef_df[coef_df["dep_var"] == dep].iloc[0]
        curves[dep] = partial_effect_curve(float(r["beta_ind_own"]), float(r["beta_ind_own_sq"]), xgrid, x_ref)

    # Higher score = better market quality:
    # lower amihud better, lower volatility better, higher price_info better.
    def z(v: np.ndarray) -> np.ndarray:
        s = np.std(v)
        if s <= 1e-12:
            return np.zeros_like(v)
        return (v - np.mean(v)) / s

    score = -z(curves["amihud_illiq"]) - z(curves["volatility"]) + z(curves["price_info"])
    best_idx = int(np.argmax(score))
    best_x = float(xgrid[best_idx])

    q95 = float(np.quantile(score, 0.95))
    good_mask = score >= q95
    good_x = xgrid[good_mask]
    good_lo = float(good_x.min()) if len(good_x) else np.nan
    good_hi = float(good_x.max()) if len(good_x) else np.nan

    out = pd.DataFrame(
        {
            "ind_own": xgrid,
            "score_total": score,
            "delta_amihud": curves["amihud_illiq"],
            "delta_volatility": curves["volatility"],
            "delta_price_info": curves["price_info"],
            "is_top5pct_score": good_mask.astype(int),
        }
    )

    fig, ax = plt.subplots(figsize=(10.8, 4.8))
    ax.plot(xgrid, score, color="#1f77b4", linewidth=2.1, label="Market quality score")
    ax.axvline(best_x, color="#d62728", linestyle="--", linewidth=1.4, label=f"Best indexing ~ {best_x:.3f}")
    if np.isfinite(good_lo) and np.isfinite(good_hi):
        ax.axvspan(good_lo, good_hi, color="#9ecae1", alpha=0.32, lw=0, label="Top 5% score range")
    ax.set_title("How Much Indexing Is Good? Nonlinear Combined Score")
    ax.set_xlabel("ind_own")
    ax.set_ylabel("Standardized composite score")
    ax.grid(axis="y", alpha=0.25, linestyle=":")
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(OUT_PLOT_SCORE, format="svg", dpi=220, bbox_inches="tight")
    plt.close(fig)

    meta = pd.DataFrame(
        [
            {"ind_own": best_x, "score_total": float(np.max(score)), "delta_amihud": np.nan, "delta_volatility": np.nan, "delta_price_info": np.nan, "is_top5pct_score": 2},
            {"ind_own": good_lo, "score_total": q95, "delta_amihud": np.nan, "delta_volatility": np.nan, "delta_price_info": np.nan, "is_top5pct_score": 3},
            {"ind_own": good_hi, "score_total": q95, "delta_amihud": np.nan, "delta_volatility": np.nan, "delta_price_info": np.nan, "is_top5pct_score": 4},
        ]
    )
    return pd.concat([out, meta], ignore_index=True)


def write_summary(coef_df: pd.DataFrame, score_df: pd.DataFrame) -> None:
    best_row = score_df[score_df["is_top5pct_score"] == 2].iloc[0]
    lo_row = score_df[score_df["is_top5pct_score"] == 3].iloc[0]
    hi_row = score_df[score_df["is_top5pct_score"] == 4].iloc[0]
    best_x = float(best_row["ind_own"])
    good_lo = float(lo_row["ind_own"])
    good_hi = float(hi_row["ind_own"])

    lines = []
    lines.append("# Nonlinearity: How Much Indexing Is Good?")
    lines.append("")
    lines.append("Model: panel FE with quadratic term")
    lines.append("- `y_it = b1*ind_own + b2*ind_own^2 + controls + firm FE + time FE`")
    lines.append("- SE: two-way clustered (firm, date)")
    lines.append("")
    lines.append("## Core answer")
    lines.append(f"- Composite optimum indexing level: `ind_own ≈ {best_x:.4f}`")
    lines.append(f"- High-quality range (top 5% score): `[{good_lo:.4f}, {good_hi:.4f}]`")
    lines.append("")
    lines.append("## Quadratic terms")
    for dep in DEP_VARS:
        r = coef_df[coef_df["dep_var"] == dep].iloc[0]
        lines.append(
            f"- `{dep}`: b1={r['beta_ind_own']:.4g} (p={r['p_ind_own']:.3g}), "
            f"b2={r['beta_ind_own_sq']:.4g} (p={r['p_ind_own_sq']:.3g}), "
            f"turning={r['turning_point']:.4f}, in_support={bool(r['turning_in_q01_q99'])}"
        )
    lines.append("")
    lines.append("Interpretation note: lower `amihud_illiq` and `volatility` are better; higher `price_info` is better.")
    OUT_SUMMARY.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA_PATH)

    coef_rows = [fit_one(df, dep) for dep in DEP_VARS]
    coef_df = pd.DataFrame(coef_rows)
    coef_df.to_csv(OUT_COEF, index=False)

    score_df = build_score_and_plot(coef_df)
    score_df.to_csv(OUT_SCORE, index=False)

    write_summary(coef_df, score_df)

    print(f"Saved: {OUT_COEF}")
    print(f"Saved: {OUT_SCORE}")
    print(f"Saved: {OUT_PLOT_SCORE}")
    print(f"Saved: {OUT_SUMMARY}")


if __name__ == "__main__":
    main()
