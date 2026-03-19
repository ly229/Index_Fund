#!/usr/bin/env python3
from __future__ import annotations

import csv
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from patsy import build_design_matrices, dmatrix
from scipy.stats import chi2

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import PercentFormatter


MAIN_PATH = Path("data/df_regression.csv")
RANK_CANDIDATES = [Path("data/Russell_3_rank.csv"), Path("Ressell_Rank/Russell_3_rank.csv")]
OUT_DIR = Path("analysis/nonlinearity/IV")

OUT_RESULTS = OUT_DIR / "iv_spline_cutoff2000_global_results.csv"
OUT_CURVES = OUT_DIR / "iv_spline_cutoff2000_global_curves.csv"
OUT_SCORE = OUT_DIR / "iv_spline_cutoff2000_global_score.csv"
OUT_SUMMARY = OUT_DIR / "iv_spline_cutoff2000_global_summary.md"
OUT_PLOT_CURVES = OUT_DIR / "iv_spline_cutoff2000_global_curves.svg"
OUT_PLOT_SCORE = OUT_DIR / "iv_spline_cutoff2000_global_score.svg"

DEP_VARS = ["amihud_illiq", "volatility", "price_info"]
RHS_BASE = ["c_firm_size", "c_dollar_vol"]
MAIN = "ind_own"

CUTOFF = 2000.0
BANDWIDTH = None
POLY_DEG = 4
KNOT_QUANTILES = [0.25, 0.50, 0.75]
SCALE_LO_Q = 0.01
SCALE_HI_Q = 0.99

DEP_META = {
    "amihud_illiq": {"title": "Illiquidity", "color": "#8c2d04"},
    "volatility": {"title": "Volatility", "color": "#08519c"},
    "price_info": {"title": "Price Informativeness", "color": "#006d2c"},
}


def parse_float(x: str):
    if x is None:
        return None
    x = x.strip()
    if x == "":
        return None
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except ValueError:
        return None


def parse_date(x: str):
    return datetime.strptime(x, "%Y-%m-%d")


def rank_year_for_panel_date(dt: datetime) -> int:
    return dt.year if dt.month >= 6 else dt.year - 1


def normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def load_rank_map():
    rank_path = next((p for p in RANK_CANDIDATES if p.exists()), None)
    if rank_path is None:
        raise FileNotFoundError("No rank file found.")
    out = {}
    with rank_path.open("r", newline="") as f:
        rd = csv.DictReader(f)
        for r in rd:
            cusip = r.get("cusip", "").strip()
            ds = r.get("date", "").strip()
            rv = parse_float(r.get("Rank", ""))
            if not cusip or not ds or rv is None:
                continue
            yr = parse_date(ds).year
            k = (cusip, yr)
            if k not in out or rv < out[k]:
                out[k] = rv
    return out


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


def compute_knots_scaled(df: pd.DataFrame, scale_lo: float, scale_hi: float) -> tuple[list[float], list[float]]:
    x = pd.to_numeric(df[MAIN], errors="coerce")
    x = x[np.isfinite(x)]
    knots_raw = [float(x.quantile(q)) for q in KNOT_QUANTILES]
    knots_raw = [min(max(k, scale_lo), scale_hi) for k in knots_raw]
    knots_scaled = [float(scale_ind_own(np.array([k]), scale_lo, scale_hi)[0]) for k in knots_raw]
    return knots_raw, knots_scaled


def spline_basis_from_scaled(x_scaled: np.ndarray, knots_scaled: list[float]):
    return dmatrix(
        "cr(x, knots=knots, lower_bound=0, upper_bound=1, constraints='center') - 1",
        {"x": x_scaled, "knots": tuple(knots_scaled)},
        return_type="dataframe",
    )


def two_way_demean(mat: np.ndarray, firm_ids: np.ndarray, time_ids: np.ndarray, tol=1e-9, max_iter=200):
    x = mat.astype(float, copy=True)
    n_firms = int(firm_ids.max()) + 1
    n_times = int(time_ids.max()) + 1

    firm_counts = np.bincount(firm_ids, minlength=n_firms).astype(float)
    time_counts = np.bincount(time_ids, minlength=n_times).astype(float)

    for _ in range(max_iter):
        prev = x.copy()
        for j in range(x.shape[1]):
            v = x[:, j]
            fsum = np.bincount(firm_ids, weights=v, minlength=n_firms)
            fmean = fsum / np.maximum(firm_counts, 1.0)
            x[:, j] = v - fmean[firm_ids]
        for j in range(x.shape[1]):
            v = x[:, j]
            tsum = np.bincount(time_ids, weights=v, minlength=n_times)
            tmean = tsum / np.maximum(time_counts, 1.0)
            x[:, j] = v - tmean[time_ids]
        x += x.mean(axis=0, keepdims=True)
        if np.max(np.abs(x - prev)) < tol:
            break
    return x


def clustered_ols(y: np.ndarray, X: np.ndarray, cluster_ids: np.ndarray):
    n, k = X.shape
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    u = y - X @ beta
    XtX_inv = np.linalg.pinv(X.T @ X)
    n_clusters = int(cluster_ids.max()) + 1
    meat = np.zeros((k, k), dtype=float)
    for g in range(n_clusters):
        idx = cluster_ids == g
        if not np.any(idx):
            continue
        s = X[idx, :].T @ u[idx]
        meat += np.outer(s, s)
    corr = 1.0
    if n_clusters > 1 and (n - k) > 0:
        corr = (n_clusters / (n_clusters - 1.0)) * ((n - 1.0) / (n - k))
    vcov = corr * (XtX_inv @ meat @ XtX_inv)
    return beta, vcov


def first_stage_joint_f(x: np.ndarray, z_excl: np.ndarray, w: np.ndarray, cluster_ids: np.ndarray):
    X = np.column_stack([z_excl, w])
    beta, vcov = clustered_ols(x, X, cluster_ids)
    q = z_excl.shape[1]
    b = beta[:q]
    V = vcov[:q, :q]
    V_inv = np.linalg.pinv(V)
    wald = float(b.T @ V_inv @ b)
    fstat = wald / q if q > 0 else float("nan")
    pval = 1.0 - chi2.cdf(wald, q) if q > 0 else float("nan")
    return fstat, pval


def iv_2sls_clustered(y: np.ndarray, x_endog: np.ndarray, z_excl: np.ndarray, w: np.ndarray, cluster_ids: np.ndarray):
    X = np.column_stack([x_endog, w])
    Z = np.column_stack([z_excl, w])

    n = X.shape[0]
    k = X.shape[1]
    Qxz = (X.T @ Z) / n
    Qzz = (Z.T @ Z) / n
    Qzz_inv = np.linalg.pinv(Qzz)
    A = Qxz @ Qzz_inv @ Qxz.T
    A_inv = np.linalg.pinv(A)
    beta = A_inv @ (Qxz @ Qzz_inv @ ((Z.T @ y) / n))

    u = y - X @ beta
    n_clusters = int(cluster_ids.max()) + 1
    l = Z.shape[1]
    S = np.zeros((l, l), dtype=float)
    for g in range(n_clusters):
        idx = cluster_ids == g
        if not np.any(idx):
            continue
        Gg = Z[idx, :].T @ u[idx]
        S += np.outer(Gg, Gg)
    S /= n

    meat = Qxz @ Qzz_inv @ S @ Qzz_inv @ Qxz.T
    vcov = A_inv @ meat @ A_inv / n
    corr = 1.0
    if n_clusters > 1 and (n - k) > 0:
        corr = (n_clusters / (n_clusters - 1.0)) * ((n - 1.0) / (n - k))
    vcov *= corr
    se = np.sqrt(np.maximum(np.diag(vcov), 0.0))
    return beta, se, vcov


def prepare(dep_var: str, rank_map: dict, knots_scaled: list[float], scale_lo: float, scale_hi: float):
    rows = []
    with MAIN_PATH.open("r", newline="") as f:
        rd = csv.DictReader(f)
        for r in rd:
            cusip = r.get("cusip", "").strip()
            ds = r.get("date", "").strip()
            if not cusip or not ds:
                continue
            y = parse_float(r.get(dep_var, ""))
            x_raw = parse_float(r.get(MAIN, ""))
            c1 = parse_float(r.get(RHS_BASE[0], ""))
            c2 = parse_float(r.get(RHS_BASE[1], ""))
            if any(v is None for v in [y, x_raw, c1, c2]):
                continue

            dt = parse_date(ds)
            ryear = rank_year_for_panel_date(dt)
            rank = rank_map.get((cusip, ryear))
            if rank is None:
                continue
            running = rank - CUTOFF
            if BANDWIDTH is not None and abs(running) > BANDWIDTH:
                continue
            z = 1.0 if rank <= CUTOFF else 0.0
            running_s = running / CUTOFF

            rows.append(
                {
                    "cusip": cusip,
                    "date": ds,
                    "y": y,
                    "x_raw": x_raw,
                    "c_firm_size": c1,
                    "c_dollar_vol": c2,
                    "running_s": running_s,
                    "z_cutoff": z,
                }
            )

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError(f"No observations available for {dep_var}.")

    df["x_scaled"] = scale_ind_own(df["x_raw"], scale_lo, scale_hi)
    basis = spline_basis_from_scaled(df["x_scaled"].to_numpy(dtype=float), knots_scaled)
    basis_cols = [f"spline_{i+1}" for i in range(basis.shape[1])]
    basis.columns = basis_cols
    df = pd.concat([df.reset_index(drop=True), basis.reset_index(drop=True)], axis=1)

    for k in range(1, POLY_DEG + 1):
        df[f"running_pow{k}"] = df["running_s"] ** k
    for k in range(POLY_DEG + 1):
        df[f"z_running_pow{k}"] = df["z_cutoff"] * (df["running_s"] ** k)

    firm_ids, _ = pd.factorize(df["cusip"], sort=True)
    time_ids, _ = pd.factorize(df["date"], sort=True)

    y = df["y"].to_numpy(dtype=float)
    x_endog = df[basis_cols].to_numpy(dtype=float)
    z_excl_cols = [f"z_running_pow{k}" for k in range(POLY_DEG + 1)]
    w_cols = RHS_BASE + [f"running_pow{k}" for k in range(1, POLY_DEG + 1)]
    z_excl = df[z_excl_cols].to_numpy(dtype=float)
    w = df[w_cols].to_numpy(dtype=float)

    M = np.column_stack([y, x_endog, z_excl, w])
    M_dm = two_way_demean(M, firm_ids.astype(int), time_ids.astype(int))
    y_dm = M_dm[:, 0]
    n_endog = x_endog.shape[1]
    n_excl = z_excl.shape[1]
    x_dm = M_dm[:, 1 : 1 + n_endog]
    z_dm = M_dm[:, 1 + n_endog : 1 + n_endog + n_excl]
    w_dm = M_dm[:, 1 + n_endog + n_excl :]

    return {
        "df": df,
        "basis_cols": basis_cols,
        "z_excl_cols": z_excl_cols,
        "w_cols": w_cols,
        "y_dm": y_dm,
        "x_dm": x_dm,
        "z_dm": z_dm,
        "w_dm": w_dm,
        "firm_ids": firm_ids.astype(int),
        "time_ids": time_ids.astype(int),
    }


def predict_curve(xgrid_raw: np.ndarray, x_ref_raw: float, coef_basis: np.ndarray, vcov_basis: np.ndarray, knots_scaled: list[float], scale_lo: float, scale_hi: float):
    design_info = spline_basis_from_scaled(np.array([0.25, 0.5, 0.75], dtype=float), knots_scaled).design_info
    grid_scaled = scale_ind_own(xgrid_raw, scale_lo, scale_hi)
    ref_scaled = scale_ind_own(np.array([x_ref_raw], dtype=float), scale_lo, scale_hi)
    grid_basis = build_design_matrices(
        [design_info],
        {"x": grid_scaled, "knots": tuple(knots_scaled)},
        return_type="dataframe",
    )[0].to_numpy(dtype=float)
    ref_basis = build_design_matrices(
        [design_info],
        {"x": ref_scaled, "knots": tuple(knots_scaled)},
        return_type="dataframe",
    )[0].to_numpy(dtype=float)
    diff_basis = grid_basis - ref_basis
    curve = diff_basis @ coef_basis
    var_curve = np.einsum("ij,jk,ik->i", diff_basis, vcov_basis, diff_basis)
    se_curve = np.sqrt(np.maximum(var_curve, 0.0))
    return curve, curve - 1.96 * se_curve, curve + 1.96 * se_curve


def run_one(dep_var: str, rank_map: dict, knots_raw: list[float], knots_scaled: list[float], scale_lo: float, scale_hi: float):
    pack = prepare(dep_var, rank_map, knots_scaled, scale_lo, scale_hi)
    beta, se, vcov = iv_2sls_clustered(pack["y_dm"], pack["x_dm"], pack["z_dm"], pack["w_dm"], pack["firm_ids"])

    n_endog = pack["x_dm"].shape[1]
    coef_basis = beta[:n_endog]
    se_basis = se[:n_endog]
    vcov_basis = vcov[:n_endog, :n_endog]

    first_stage_rows = []
    for j, col in enumerate(pack["basis_cols"]):
        fstat, pval = first_stage_joint_f(pack["x_dm"][:, j], pack["z_dm"], pack["w_dm"], pack["firm_ids"])
        first_stage_rows.append((col, fstat, pval))

    xraw = pack["df"]["x_raw"].to_numpy(dtype=float)
    q01 = float(np.quantile(xraw, 0.01))
    q50 = float(np.quantile(xraw, 0.50))
    q99 = float(np.quantile(xraw, 0.99))
    xgrid = np.linspace(max(q01, scale_lo), min(q99, scale_hi), 400)
    curve, ci_lo, ci_hi = predict_curve(xgrid, q50, coef_basis, vcov_basis, knots_scaled, scale_lo, scale_hi)

    if dep_var == "price_info":
        best_idx = int(np.argmax(curve))
    else:
        best_idx = int(np.argmin(curve))

    result = {
        "dep_var": dep_var,
        "nobs": int(pack["df"].shape[0]),
        "n_firms": int(pack["df"]["cusip"].nunique()),
        "n_dates": int(pack["df"]["date"].nunique()),
        "cutoff": CUTOFF,
        "bandwidth": "" if BANDWIDTH is None else BANDWIDTH,
        "poly_deg": POLY_DEG,
        "scale_lo": scale_lo,
        "scale_hi": scale_hi,
        "ind_own_q01": q01,
        "ind_own_q50": q50,
        "ind_own_q99": q99,
        "best_ind_own_for_depvar": float(xgrid[best_idx]),
        "best_curve_value": float(curve[best_idx]),
        "knot_1": knots_raw[0],
        "knot_2": knots_raw[1],
        "knot_3": knots_raw[2],
        "excluded_instruments": ";".join(pack["z_excl_cols"]),
        "controls": ";".join(pack["w_cols"]),
    }
    for j, col in enumerate(pack["basis_cols"]):
        t = coef_basis[j] / se_basis[j] if se_basis[j] > 0 else float("nan")
        p = 2.0 * (1.0 - normal_cdf(abs(t))) if np.isfinite(t) else float("nan")
        result[f"beta_{col}"] = float(coef_basis[j])
        result[f"se_{col}"] = float(se_basis[j])
        result[f"t_{col}"] = float(t)
        result[f"p_{col}"] = float(p)
        result[f"first_stage_F_{col}"] = float(first_stage_rows[j][1])
        result[f"first_stage_P_{col}"] = float(first_stage_rows[j][2])
    result["min_first_stage_F"] = float(min(v for _, v, _ in first_stage_rows))

    curves = pd.DataFrame(
        {
            "dep_var": dep_var,
            "ind_own": xgrid,
            "curve_value": curve,
            "ci_lo": ci_lo,
            "ci_hi": ci_hi,
        }
    )
    return result, curves


def zscore(v: np.ndarray) -> np.ndarray:
    s = np.std(v)
    if s <= 1e-12:
        return np.zeros_like(v)
    return (v - np.mean(v)) / s


def build_score(curves_df: pd.DataFrame, results_df: pd.DataFrame) -> pd.DataFrame:
    x_lo = float(results_df["ind_own_q01"].max())
    x_hi = float(results_df["ind_own_q99"].min())
    xgrid = np.linspace(x_lo, x_hi, 400)
    aligned = {}
    for dep in DEP_VARS:
        d = curves_df[curves_df["dep_var"] == dep]
        aligned[dep] = np.interp(xgrid, d["ind_own"], d["curve_value"])

    score = -zscore(aligned["amihud_illiq"]) - zscore(aligned["volatility"]) + zscore(aligned["price_info"])
    best_idx = int(np.argmax(score))
    q95 = float(np.quantile(score, 0.95))
    good_mask = score >= q95
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
            {"ind_own": float(xgrid[best_idx]), "score_total": float(score[best_idx]), "delta_amihud": np.nan, "delta_volatility": np.nan, "delta_price_info": np.nan, "is_top5pct_score": 2},
            {"ind_own": float(xgrid[good_mask].min()), "score_total": q95, "delta_amihud": np.nan, "delta_volatility": np.nan, "delta_price_info": np.nan, "is_top5pct_score": 3},
            {"ind_own": float(xgrid[good_mask].max()), "score_total": q95, "delta_amihud": np.nan, "delta_volatility": np.nan, "delta_price_info": np.nan, "is_top5pct_score": 4},
        ]
    )
    return pd.concat([out, meta], ignore_index=True)


def set_style() -> None:
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
        }
    )


def style_axis(ax):
    ax.grid(axis="y", color="#d9d9d9", linestyle=(0, (2, 2)), linewidth=0.8)
    ax.tick_params(axis="both", which="major", length=4, width=0.8, color="#333333")


def plot_curves(curves_df: pd.DataFrame, knots_raw: list[float]) -> None:
    set_style()
    fig = plt.figure(figsize=(12.2, 8.6))
    panel_w = 0.31
    panel_h = 0.31
    top_y = 0.57
    bottom_y = 0.16
    left_x = 0.10
    right_x = 0.59
    center_x = 0.345
    axes = [
        fig.add_axes([left_x, top_y, panel_w, panel_h]),
        fig.add_axes([right_x, top_y, panel_w, panel_h]),
        fig.add_axes([center_x, bottom_y, panel_w, panel_h]),
    ]
    panel_marks = {"amihud_illiq": "a)", "volatility": "b)", "price_info": "c)"}
    support_lo = float(curves_df["ind_own"].min())
    support_hi = float(curves_df["ind_own"].max())

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
        for knot_x in knots_raw:
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
    axes[0].set_ylabel("IV spline effect relative to median `ind_own`")
    axes[2].set_ylabel("IV spline effect relative to median `ind_own`")
    legend_handles = [
        Line2D([0], [0], color="#4d4d4d", linewidth=2.2, label="IV spline fit"),
        Patch(facecolor="#bdbdbd", edgecolor="none", alpha=0.25, label="95% confidence band"),
        Line2D([0], [0], color="#c7c7c7", linestyle=(0, (4, 4)), linewidth=1.0, label="Interior knots"),
        Line2D([0], [0], color="#9e9e9e", linestyle=(0, (2, 3)), linewidth=1.0, label="Common support"),
    ]
    fig.suptitle("IV Spline Nonlinearity Across Market-Quality Measures", fontsize=14, y=0.955)
    fig.legend(handles=legend_handles, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.045))
    fig.savefig(OUT_PLOT_CURVES, format="svg", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_score(score_df: pd.DataFrame) -> None:
    set_style()
    score_main = score_df[score_df["is_top5pct_score"] <= 1].copy()
    best_row = score_df[score_df["is_top5pct_score"] == 2].iloc[0]
    lo_row = score_df[score_df["is_top5pct_score"] == 3].iloc[0]
    hi_row = score_df[score_df["is_top5pct_score"] == 4].iloc[0]
    support_lo = float(score_main["ind_own"].min())
    support_hi = float(score_main["ind_own"].max())
    score_floor = float(score_main["score_total"].min())
    peak_y = float(best_row["score_total"])

    fig, ax = plt.subplots(figsize=(11.2, 5.9))
    ax.fill_between(score_main["ind_own"], score_main["score_total"], score_floor - 0.12, color="#9ecae1", alpha=0.18, zorder=1)
    ax.axvspan(float(lo_row["ind_own"]), float(hi_row["ind_own"]), color="#dbe9f6", alpha=0.9, lw=0, zorder=0)
    ax.plot(score_main["ind_own"], score_main["score_total"], color="#0b3c5d", linewidth=2.6, zorder=3)
    ax.axvline(float(best_row["ind_own"]), color="#a50f15", linestyle=(0, (6, 3)), linewidth=1.6, zorder=4)
    ax.axhline(0.0, color="#666666", linestyle=(0, (1, 2)), linewidth=1.0, zorder=1)
    ax.set_xlim(support_lo, support_hi)
    ax.set_title("IV Spline Market Quality Score", pad=14)
    ax.text(
        0.0,
        1.04,
        "Global Russell cutoff-IV (cutoff = 2000) with firm and time fixed effects",
        transform=ax.transAxes,
        fontsize=10.1,
        color="#4d4d4d",
    )
    ax.set_xlabel("Index ownership share (`ind_own`)")
    ax.set_ylabel("Composite market-quality score")
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    style_axis(ax)
    ax.scatter([float(best_row["ind_own"])], [peak_y], color="#a50f15", s=36, zorder=5)
    ax.annotate(
        f"Optimum = {float(best_row['ind_own']):.3f}",
        xy=(float(best_row["ind_own"]), peak_y),
        xytext=(14, 16),
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


def write_summary(results_df: pd.DataFrame, score_df: pd.DataFrame, knots_raw: list[float]) -> None:
    best_row = score_df[score_df["is_top5pct_score"] == 2].iloc[0]
    lo_row = score_df[score_df["is_top5pct_score"] == 3].iloc[0]
    hi_row = score_df[score_df["is_top5pct_score"] == 4].iloc[0]

    lines = [
        "# IV Spline Nonlinearity Results",
        "",
        "Specification:",
        "- 2SLS with endogenous natural cubic spline basis in `ind_own`",
        f"- Preferred IV design: global sample, cutoff={int(CUTOFF)}, bandwidth=None",
        f"- Excluded instruments: `z_cutoff * running^k`, `k=0..{POLY_DEG}`",
        f"- Controls: `c_firm_size`, `c_dollar_vol`, `running^k`, `k=1..{POLY_DEG}`",
        "- FE: firm + date (via two-way demeaning)",
        "- SE: firm-clustered",
        f"- Spline knots in raw `ind_own`: `{', '.join(f'{k:.4f}' for k in knots_raw)}`",
        "",
        "## Core answer",
        f"- Composite optimum indexing level: `ind_own ≈ {float(best_row['ind_own']):.4f}`",
        f"- High-quality range (top 5% score): `[{float(lo_row['ind_own']):.4f}, {float(hi_row['ind_own']):.4f}]`",
        "",
        "## Outcome summaries",
    ]
    for dep in DEP_VARS:
        r = results_df[results_df["dep_var"] == dep].iloc[0]
        lines.append(
            f"- `{dep}`: best `ind_own ≈ {float(r['best_ind_own_for_depvar']):.4f}`, "
            f"min first-stage F across spline terms = {float(r['min_first_stage_F']):.2f}, n = {int(r['nobs'])}"
        )
    lines.append("")
    lines.append("Interpretation note: this is the causal spline extension of the rank-cutoff IV design. If first-stage strength is weak for some spline terms, interpret the curve shape cautiously.")
    OUT_SUMMARY.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df_main = pd.read_csv(MAIN_PATH, usecols=[MAIN])
    scale_lo, scale_hi = compute_scale_bounds(df_main)
    knots_raw, knots_scaled = compute_knots_scaled(df_main, scale_lo, scale_hi)
    rank_map = load_rank_map()

    results = []
    curve_parts = []
    for dep in DEP_VARS:
        r, curves = run_one(dep, rank_map, knots_raw, knots_scaled, scale_lo, scale_hi)
        results.append(r)
        curve_parts.append(curves)

    results_df = pd.DataFrame(results)
    curves_df = pd.concat(curve_parts, ignore_index=True)
    score_df = build_score(curves_df, results_df)

    results_df.to_csv(OUT_RESULTS, index=False)
    curves_df.to_csv(OUT_CURVES, index=False)
    score_df.to_csv(OUT_SCORE, index=False)
    plot_curves(curves_df, knots_raw)
    plot_score(score_df)
    write_summary(results_df, score_df, knots_raw)

    print(f"Saved: {OUT_RESULTS}")
    print(f"Saved: {OUT_CURVES}")
    print(f"Saved: {OUT_SCORE}")
    print(f"Saved: {OUT_PLOT_CURVES}")
    print(f"Saved: {OUT_PLOT_SCORE}")
    print(f"Saved: {OUT_SUMMARY}")


if __name__ == "__main__":
    main()
