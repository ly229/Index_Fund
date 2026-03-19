#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS
from patsy import build_design_matrices, dmatrix

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import PercentFormatter


DATA_PATH = Path("data/df_regression.csv")
OUT_DIR = Path("analysis/nonlinearity")

DEP_VARS = ["amihud_illiq", "volatility", "price_info"]
RHS_BASE = ["c_firm_size", "c_dollar_vol"]
MAIN = "ind_own"

KNOT_QUANTILES = [0.25, 0.50, 0.75]

OUT_COEF = OUT_DIR / "spline_panel_results.csv"
OUT_SCORE = OUT_DIR / "spline_indexing_score_grid.csv"
OUT_SUMMARY = OUT_DIR / "spline_indexing_summary.md"
OUT_PLOT_SCORE = OUT_DIR / "spline_good_range_score.svg"
OUT_PLOT_CURVES = OUT_DIR / "spline_relationship_all3.svg"

DEP_META = {
    "amihud_illiq": {"title": "Illiquidity", "color": "#8c2d04"},
    "volatility": {"title": "Volatility", "color": "#08519c"},
    "price_info": {"title": "Price Informativeness", "color": "#006d2c"},
}

SCALE_LO_Q = 0.01
SCALE_HI_Q = 0.99


def compute_global_knots(df: pd.DataFrame) -> list[float]:
    x = pd.to_numeric(df[MAIN], errors="coerce")
    x = x[np.isfinite(x)]
    return [float(x.quantile(q)) for q in KNOT_QUANTILES]


def compute_scale_bounds(df: pd.DataFrame) -> tuple[float, float]:
    x = pd.to_numeric(df[MAIN], errors="coerce")
    x = x[np.isfinite(x)]
    lo = float(x.quantile(SCALE_LO_Q))
    hi = float(x.quantile(SCALE_HI_Q))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        raise ValueError("Invalid scaling bounds for ind_own.")
    return lo, hi


def scale_ind_own(x: pd.Series | np.ndarray, lo: float, hi: float) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    clipped = np.clip(arr, lo, hi)
    return (clipped - lo) / (hi - lo)


def add_spline_basis(
    d: pd.DataFrame,
    knots_scaled: list[float],
    lower_bound: float,
    upper_bound: float,
) -> tuple[pd.DataFrame, list[str], object]:
    basis = dmatrix(
        "cr(x, knots=knots, lower_bound=lower_bound, upper_bound=upper_bound, constraints='center') - 1",
        {
            "x": d["ind_own_scaled"].to_numpy(dtype=float),
            "knots": tuple(knots_scaled),
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
        },
        return_type="dataframe",
    )
    basis_cols = [f"spline_{i+1}" for i in range(basis.shape[1])]
    basis.columns = basis_cols
    out = pd.concat([d.reset_index(drop=True), basis.reset_index(drop=True)], axis=1)
    return out, basis_cols, basis.design_info


def prep(
    df: pd.DataFrame,
    dep_var: str,
    knots_scaled: list[float],
    scale_lo: float,
    scale_hi: float,
) -> tuple[pd.DataFrame, list[str], object]:
    cols = ["cusip", "date", MAIN, dep_var] + RHS_BASE
    d = df[cols].copy()
    d["date"] = pd.to_datetime(d["date"])
    for c in [dep_var, MAIN] + RHS_BASE:
        d[c] = pd.to_numeric(d[c], errors="coerce")
        d.loc[~np.isfinite(d[c]), c] = np.nan
    d = d.dropna(subset=[dep_var, MAIN] + RHS_BASE)
    d["ind_own_scaled"] = scale_ind_own(d[MAIN], scale_lo, scale_hi)
    d, basis_cols, design_info = add_spline_basis(d, knots_scaled, 0.0, 1.0)
    d = d.set_index(["cusip", "date"]).sort_index()
    return d, basis_cols, design_info


def predict_spline_component(
    xgrid: np.ndarray,
    x_ref: float,
    basis_cols: list[str],
    design_info: object,
    coef: pd.Series,
    cov: pd.DataFrame,
    knots_scaled: list[float],
    scale_lo: float,
    scale_hi: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xgrid_scaled = scale_ind_own(xgrid, scale_lo, scale_hi)
    x_ref_scaled = scale_ind_own(np.array([x_ref], dtype=float), scale_lo, scale_hi)
    grid_basis = build_design_matrices(
        [design_info],
        {
            "x": xgrid_scaled,
            "knots": tuple(knots_scaled),
            "lower_bound": 0.0,
            "upper_bound": 1.0,
        },
        return_type="dataframe",
    )[0]
    ref_basis = build_design_matrices(
        [design_info],
        {
            "x": x_ref_scaled,
            "knots": tuple(knots_scaled),
            "lower_bound": 0.0,
            "upper_bound": 1.0,
        },
        return_type="dataframe",
    )[0]
    beta = coef.loc[basis_cols].to_numpy(dtype=float)
    cov_basis = cov.loc[basis_cols, basis_cols].to_numpy(dtype=float)
    grid_arr = grid_basis.to_numpy(dtype=float)
    ref_arr = ref_basis.to_numpy(dtype=float)
    diff_basis = grid_arr - ref_arr
    g = diff_basis @ beta
    var_g = np.einsum("ij,jk,ik->i", diff_basis, cov_basis, diff_basis)
    se_g = np.sqrt(np.maximum(var_g, 0.0))
    ci_lo = g - 1.96 * se_g
    ci_hi = g + 1.96 * se_g
    return g, ci_lo, ci_hi


def fit_one(
    df: pd.DataFrame,
    dep_var: str,
    knots_scaled: list[float],
    scale_lo: float,
    scale_hi: float,
) -> tuple[dict, pd.DataFrame]:
    d, basis_cols, design_info = prep(df, dep_var, knots_scaled, scale_lo, scale_hi)
    formula = f"{dep_var} ~ 0 + {' + '.join(basis_cols)} + {' + '.join(RHS_BASE)} + EntityEffects + TimeEffects"
    model = PanelOLS.from_formula(formula, data=d, drop_absorbed=True)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        fit = model.fit(cov_type="clustered", cluster_entity=True, cluster_time=True)

    x = d[MAIN]
    q01, q05, q10, q50, q90, q95, q99 = [float(x.quantile(q)) for q in [0.01, 0.05, 0.10, 0.50, 0.90, 0.95, 0.99]]
    xmin, xmax = float(x.min()), float(x.max())
    xgrid = np.linspace(max(q01, scale_lo), min(q99, scale_hi), 400)
    curve, ci_lo, ci_hi = predict_spline_component(
        xgrid,
        q50,
        basis_cols,
        design_info,
        fit.params,
        fit.cov,
        knots_scaled,
        scale_lo,
        scale_hi,
    )

    if dep_var == "price_info":
        best_idx = int(np.argmax(curve))
    else:
        best_idx = int(np.argmin(curve))
    best_x = float(xgrid[best_idx])
    best_y = float(curve[best_idx])

    row = {
        "dep_var": dep_var,
        "nobs": int(fit.nobs),
        "entities": int(d.index.get_level_values(0).nunique()),
        "time_periods": int(d.index.get_level_values(1).nunique()),
        "ind_own_min": xmin,
        "ind_own_q01": q01,
        "ind_own_q05": q05,
        "ind_own_q10": q10,
        "ind_own_q50": q50,
        "ind_own_q90": q90,
        "ind_own_q95": q95,
        "ind_own_q99": q99,
        "ind_own_max": xmax,
        "best_ind_own_for_depvar": best_x,
        "best_curve_value": best_y,
        "rsquared_within": float(getattr(fit, "rsquared_within", np.nan)),
    }
    for col in basis_cols:
        row[f"beta_{col}"] = float(fit.params[col])
        row[f"se_{col}"] = float(fit.std_errors[col])
        row[f"p_{col}"] = float(fit.pvalues[col])

    curve_df = pd.DataFrame(
        {
            "dep_var": dep_var,
            "ind_own": xgrid,
            "curve_value": curve,
            "ci_lo": ci_lo,
            "ci_hi": ci_hi,
        }
    )
    return row, curve_df


def z(v: np.ndarray) -> np.ndarray:
    s = np.std(v)
    if s <= 1e-12:
        return np.zeros_like(v)
    return (v - np.mean(v)) / s


def build_score(curves_df: pd.DataFrame, coef_df: pd.DataFrame) -> pd.DataFrame:
    x_lo = float(coef_df["ind_own_q01"].max())
    x_hi = float(coef_df["ind_own_q99"].min())
    xgrid = np.linspace(x_lo, x_hi, 400)
    aligned = {}
    for dep in DEP_VARS:
        d = curves_df[curves_df["dep_var"] == dep]
        aligned[dep] = np.interp(xgrid, d["ind_own"], d["curve_value"])

    score = -z(aligned["amihud_illiq"]) - z(aligned["volatility"]) + z(aligned["price_info"])
    best_idx = int(np.argmax(score))
    best_x = float(xgrid[best_idx])
    best_score = float(score[best_idx])
    q95 = float(np.quantile(score, 0.95))
    good_mask = score >= q95
    good_lo = float(xgrid[good_mask].min())
    good_hi = float(xgrid[good_mask].max())

    out = pd.DataFrame(
        {
            "ind_own": xgrid,
            "score_total": score,
            "delta_amihud": aligned["amihud_illiq"],
            "delta_volatility": aligned["volatility"],
            "delta_price_info": aligned["price_info"],
            "is_top5pct_score": good_mask.astype(int),
        }
    )
    meta = pd.DataFrame(
        [
            {"ind_own": best_x, "score_total": best_score, "delta_amihud": np.nan, "delta_volatility": np.nan, "delta_price_info": np.nan, "is_top5pct_score": 2},
            {"ind_own": good_lo, "score_total": q95, "delta_amihud": np.nan, "delta_volatility": np.nan, "delta_price_info": np.nan, "is_top5pct_score": 3},
            {"ind_own": good_hi, "score_total": q95, "delta_amihud": np.nan, "delta_volatility": np.nan, "delta_price_info": np.nan, "is_top5pct_score": 4},
        ]
    )
    return pd.concat([out, meta], ignore_index=True)


def set_research_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["STIX Two Text", "Times New Roman", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#333333",
            "axes.linewidth": 0.8,
            "axes.titlesize": 13,
            "axes.titleweight": "semibold",
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 9.5,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def style_axis(ax: plt.Axes) -> None:
    ax.grid(axis="y", color="#d9d9d9", linestyle=(0, (2, 2)), linewidth=0.8)
    ax.tick_params(axis="both", which="major", length=4, width=0.8, color="#333333")
    ax.spines["left"].set_color("#333333")
    ax.spines["bottom"].set_color("#333333")


def plot_outputs(score_df: pd.DataFrame, curves_df: pd.DataFrame, knots: list[float]) -> None:
    set_research_style()
    best_row = score_df[score_df["is_top5pct_score"] == 2].iloc[0]
    lo_row = score_df[score_df["is_top5pct_score"] == 3].iloc[0]
    hi_row = score_df[score_df["is_top5pct_score"] == 4].iloc[0]

    score_main = score_df[score_df["is_top5pct_score"] <= 1].copy()
    support_lo = float(score_main["ind_own"].min())
    support_hi = float(score_main["ind_own"].max())

    fig, ax = plt.subplots(figsize=(10.8, 5.6))
    ax.axvspan(float(lo_row["ind_own"]), float(hi_row["ind_own"]), color="#dbe9f6", alpha=0.8, lw=0, zorder=0)
    ax.plot(score_main["ind_own"], score_main["score_total"], color="#0b3c5d", linewidth=2.6, zorder=3)
    ax.fill_between(score_main["ind_own"], score_main["score_total"], score_main["score_total"].min() - 0.15, color="#6baed6", alpha=0.16, zorder=1)
    ax.axvline(float(best_row["ind_own"]), color="#a50f15", linestyle=(0, (6, 3)), linewidth=1.6, zorder=4)
    ax.axhline(0.0, color="#666666", linestyle=(0, (1, 2)), linewidth=1.0, zorder=2)
    ax.set_xlim(support_lo, support_hi)
    ax.set_title("Spline-Based Market Quality Score", pad=14)
    ax.text(0.0, 1.04, "Panel FE with firm and quarter fixed effects", transform=ax.transAxes, fontsize=10, color="#4d4d4d")
    ax.set_xlabel("Index ownership share (`ind_own`)")
    ax.set_ylabel("Composite market-quality score")
    style_axis(ax)

    peak_y = float(best_row["score_total"])
    ax.scatter([float(best_row["ind_own"])], [peak_y], color="#a50f15", s=34, zorder=5)
    ax.annotate(
        f"Optimum = {float(best_row['ind_own']):.3f}",
        xy=(float(best_row["ind_own"]), peak_y),
        xytext=(16, 16),
        textcoords="offset points",
        fontsize=10,
        color="#7f0000",
        arrowprops={"arrowstyle": "-", "color": "#7f0000", "lw": 0.9},
    )
    ax.text(
        0.985,
        0.06,
        f"Top 5% range: [{float(lo_row['ind_own']):.3f}, {float(hi_row['ind_own']):.3f}]",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9.5,
        color="#274b6d",
        bbox={"boxstyle": "round,pad=0.28", "facecolor": "white", "edgecolor": "#bdbdbd", "linewidth": 0.8},
    )

    fig.tight_layout()
    fig.savefig(OUT_PLOT_SCORE, format="svg", dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig = plt.figure(figsize=(12.2, 8.8))
    panel_w = 0.31
    panel_h = 0.31
    top_y = 0.57
    bottom_y = 0.15
    left_x = 0.10
    right_x = 0.59
    center_x = 0.345
    axes = [
        fig.add_axes([left_x, top_y, panel_w, panel_h]),
        fig.add_axes([right_x, top_y, panel_w, panel_h]),
        fig.add_axes([center_x, bottom_y, panel_w, panel_h]),
    ]

    panel_marks = {"amihud_illiq": "a)", "volatility": "b)", "price_info": "c)"}
    for ax, dep in zip(axes, DEP_VARS):
        d = curves_df[curves_df["dep_var"] == dep]
        meta = DEP_META[dep]
        ymin = float(min(d["ci_lo"].min(), d["curve_value"].min()))
        ymax = float(max(d["ci_hi"].max(), d["curve_value"].max()))
        pad = 0.08 * (ymax - ymin) if ymax > ymin else 0.05
        if dep == "price_info":
            best_idx = int(np.argmax(d["curve_value"].to_numpy(dtype=float)))
        else:
            best_idx = int(np.argmin(d["curve_value"].to_numpy(dtype=float)))
        best_x = float(d.iloc[best_idx]["ind_own"])
        best_y = float(d.iloc[best_idx]["curve_value"])

        ax.axvspan(support_lo, support_hi, color="#fbfbfb", alpha=1.0, lw=0, zorder=-2)
        ax.fill_between(d["ind_own"], d["ci_lo"], d["ci_hi"], color=meta["color"], alpha=0.14, lw=0, zorder=1)
        ax.plot(d["ind_own"], d["curve_value"], color=meta["color"], linewidth=2.5, zorder=3)
        ax.axhline(0.0, color="#666666", linestyle=(0, (1, 2)), linewidth=1.0, zorder=2)
        ax.axvline(support_lo, color="#9e9e9e", linestyle=(0, (2, 3)), linewidth=0.9, zorder=0)
        ax.axvline(support_hi, color="#9e9e9e", linestyle=(0, (2, 3)), linewidth=0.9, zorder=0)
        for knot_x in knots:
            ax.axvline(knot_x, color="#c7c7c7", linestyle=(0, (4, 4)), linewidth=0.9, zorder=0)
        ax.scatter([best_x], [best_y], color=meta["color"], s=28, edgecolor="white", linewidth=0.7, zorder=4)
        ax.annotate(
            f"{best_x:.1%}",
            xy=(best_x, best_y),
            xytext=(8, -16 if dep == "price_info" else 12),
            textcoords="offset points",
            fontsize=9.2,
            color=meta["color"],
            bbox={"boxstyle": "round,pad=0.18", "facecolor": "white", "edgecolor": "#d0d0d0", "linewidth": 0.7},
        )
        ax.set_ylim(ymin - pad, ymax + pad)
        ax.set_title(f"{panel_marks[dep]} {meta['title']}", loc="center", pad=8)
        ax.set_xlabel("Index ownership share")
        ax.set_xlim(support_lo, support_hi)
        ax.xaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
        style_axis(ax)
    axes[0].set_ylabel("Predicted change relative to median `ind_own`")
    axes[2].set_ylabel("Predicted change relative to median `ind_own`")
    legend_handles = [
        Line2D([0], [0], color="#4d4d4d", linewidth=2.2, label="Spline fit"),
        Patch(facecolor="#bdbdbd", edgecolor="none", alpha=0.25, label="95% confidence band"),
        Line2D([0], [0], color="#c7c7c7", linestyle=(0, (4, 4)), linewidth=1.0, label="Interior knots"),
        Line2D([0], [0], color="#9e9e9e", linestyle=(0, (2, 3)), linewidth=1.0, label="Common support"),
    ]
    fig.suptitle("Spline Nonlinearity Across Market-Quality Measures", fontsize=14, y=0.955)
    fig.legend(handles=legend_handles, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.045))
    fig.savefig(OUT_PLOT_CURVES, format="svg", dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_summary(coef_df: pd.DataFrame, score_df: pd.DataFrame, knots: list[float]) -> None:
    best_row = score_df[score_df["is_top5pct_score"] == 2].iloc[0]
    lo_row = score_df[score_df["is_top5pct_score"] == 3].iloc[0]
    hi_row = score_df[score_df["is_top5pct_score"] == 4].iloc[0]

    knot_pct = ", ".join(f"{int(q * 100)}%" for q in KNOT_QUANTILES)
    lines = [
        "# Spline Nonlinearity: How Much Indexing Is Good?",
        "",
        "Model: panel FE with natural cubic spline in `ind_own`",
        "- `y_it = f(ind_own_it) + controls + firm FE + time FE`",
        f"- `f(.)`: natural cubic spline with interior knots at the full-sample `ind_own` quantiles {knot_pct}",
        f"- Knot values: `{', '.join(f'{k:.4f}' for k in knots)}`",
        "- SE: two-way clustered (firm, date)",
        "",
        "## Core answer",
        f"- Composite optimum indexing level: `ind_own ≈ {float(best_row['ind_own']):.4f}`",
        f"- High-quality range (top 5% score): `[{float(lo_row['ind_own']):.4f}, {float(hi_row['ind_own']):.4f}]`",
        "",
        "## Outcome-specific optima",
    ]
    for dep in DEP_VARS:
        r = coef_df[coef_df["dep_var"] == dep].iloc[0]
        lines.append(
            f"- `{dep}`: best `ind_own ≈ {float(r['best_ind_own_for_depvar']):.4f}`, "
            f"within R^2 = {float(r['rsquared_within']):.4f}, n = {int(r['nobs'])}"
        )
    lines.extend(
        [
            "",
            "Interpretation note: lower `amihud_illiq` and `volatility` are better; higher `price_info` is better.",
            "This spline specification is descriptive non-IV evidence. It is more flexible than the quadratic benchmark and is useful for checking whether the implied optimum is driven by functional-form restrictions.",
        ]
    )
    OUT_SUMMARY.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA_PATH)
    scale_lo, scale_hi = compute_scale_bounds(df)
    knots = compute_global_knots(df)
    knots = [min(max(k, scale_lo), scale_hi) for k in knots]
    knots_scaled = [float(scale_ind_own(np.array([k]), scale_lo, scale_hi)[0]) for k in knots]

    coef_rows = []
    curve_parts = []
    for dep in DEP_VARS:
        row, curve_df = fit_one(df, dep, knots_scaled, scale_lo, scale_hi)
        coef_rows.append(row)
        curve_parts.append(curve_df)

    coef_df = pd.DataFrame(coef_rows)
    curves_df = pd.concat(curve_parts, ignore_index=True)
    score_df = build_score(curves_df, coef_df)

    coef_df.to_csv(OUT_COEF, index=False)
    score_df.to_csv(OUT_SCORE, index=False)
    plot_outputs(score_df, curves_df, knots)
    write_summary(coef_df, score_df, knots)

    print(f"Saved: {OUT_COEF}")
    print(f"Saved: {OUT_SCORE}")
    print(f"Saved: {OUT_PLOT_SCORE}")
    print(f"Saved: {OUT_PLOT_CURVES}")
    print(f"Saved: {OUT_SUMMARY}")


if __name__ == "__main__":
    main()
