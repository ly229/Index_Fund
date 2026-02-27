#!/usr/bin/env python3
from __future__ import annotations

import csv
import math
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.stats import chi2


MAIN_PATH = Path("data/df_regression.csv")
RANK_CANDIDATES = [Path("data/Russell_3_rank.csv"), Path("Ressell_Rank/Russell_3_rank.csv")]
OUT_DIR = Path("analysis/nonlinearity")

OUT_RESULTS = OUT_DIR / "iv_quadratic_results.csv"
OUT_SUMMARY = OUT_DIR / "iv_quadratic_summary.md"

DEP_VARS = ["amihud_illiq", "volatility", "price_info"]

# Keep IV design aligned with existing rank-cutoff IV analysis.
CUTOFF = 1000.0
BANDWIDTH = 150.0
POLY_DEG = 2


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
        raise FileNotFoundError("No rank file found. Checked: " + ", ".join(str(p) for p in RANK_CANDIDATES))
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


def two_way_demean(mat: np.ndarray, firm_ids: np.ndarray, time_ids: np.ndarray, tol=1e-9, max_iter=200):
    x = mat.astype(float, copy=True)
    n, k = x.shape
    n_firms = int(firm_ids.max()) + 1
    n_times = int(time_ids.max()) + 1

    firm_counts = np.bincount(firm_ids, minlength=n_firms).astype(float)
    time_counts = np.bincount(time_ids, minlength=n_times).astype(float)

    for _ in range(max_iter):
        prev = x.copy()
        for j in range(k):
            v = x[:, j]
            fsum = np.bincount(firm_ids, weights=v, minlength=n_firms)
            fmean = fsum / np.maximum(firm_counts, 1.0)
            x[:, j] = v - fmean[firm_ids]
        for j in range(k):
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


def prepare(dep_var: str, rank_map: dict):
    y_vals = []
    x1_vals = []
    x2_vals = []
    z_rows = []
    w_vals = []
    firm_ids = []
    time_ids = []

    firm_to_id = {}
    time_to_id = {}

    with MAIN_PATH.open("r", newline="") as f:
        rd = csv.DictReader(f)
        for r in rd:
            cusip = r.get("cusip", "").strip()
            ds = r.get("date", "").strip()
            if not cusip or not ds:
                continue

            y = parse_float(r.get(dep_var, ""))
            x1 = parse_float(r.get("ind_own", ""))
            c1 = parse_float(r.get("c_firm_size", ""))
            c2 = parse_float(r.get("c_dollar_vol", ""))
            if any(v is None for v in [y, x1, c1, c2]):
                continue
            x2 = x1 * x1

            dt = parse_date(ds)
            ryear = rank_year_for_panel_date(dt)
            rank = rank_map.get((cusip, ryear))
            if rank is None:
                continue
            running = rank - CUTOFF
            if abs(running) > BANDWIDTH:
                continue

            running_n = running / BANDWIDTH
            z = 1.0 if rank <= CUTOFF else 0.0
            z_excl_row = [z * (running_n ** k) for k in range(POLY_DEG + 1)]
            w_row = [running_n ** k for k in range(1, POLY_DEG + 1)]

            if cusip not in firm_to_id:
                firm_to_id[cusip] = len(firm_to_id)
            if ds not in time_to_id:
                time_to_id[ds] = len(time_to_id)

            y_vals.append(y)
            x1_vals.append(x1)
            x2_vals.append(x2)
            z_rows.append(z_excl_row)
            w_vals.append((c1, c2, *w_row))
            firm_ids.append(firm_to_id[cusip])
            time_ids.append(time_to_id[ds])

    y = np.asarray(y_vals, dtype=float)
    x_endog = np.column_stack([np.asarray(x1_vals, dtype=float), np.asarray(x2_vals, dtype=float)])
    z_excl = np.asarray(z_rows, dtype=float)
    w = np.asarray(w_vals, dtype=float)
    firm_ids = np.asarray(firm_ids, dtype=int)
    time_ids = np.asarray(time_ids, dtype=int)

    mask = (
        np.isfinite(y)
        & np.all(np.isfinite(x_endog), axis=1)
        & np.all(np.isfinite(z_excl), axis=1)
        & np.all(np.isfinite(w), axis=1)
    )
    return y[mask], x_endog[mask], z_excl[mask], w[mask], firm_ids[mask], time_ids[mask]


def run_one(dep_var: str, rank_map: dict):
    y, x_endog, z_excl, w, firm_ids, time_ids = prepare(dep_var, rank_map)
    nobs = y.shape[0]

    M = np.column_stack([y, x_endog, z_excl, w])
    M_dm = two_way_demean(M, firm_ids, time_ids)
    y_dm = M_dm[:, 0]
    x_dm = M_dm[:, 1:3]
    q = z_excl.shape[1]
    z_dm = M_dm[:, 3 : 3 + q]
    w_dm = M_dm[:, 3 + q :]

    beta, se, vcov = iv_2sls_clustered(y_dm, x_dm, z_dm, w_dm, firm_ids)

    b1, b2 = float(beta[0]), float(beta[1])
    s1, s2 = float(se[0]), float(se[1])
    t1 = b1 / s1 if s1 > 0 else float("nan")
    t2 = b2 / s2 if s2 > 0 else float("nan")
    p1 = 2.0 * (1.0 - normal_cdf(abs(t1))) if np.isfinite(t1) else float("nan")
    p2 = 2.0 * (1.0 - normal_cdf(abs(t2))) if np.isfinite(t2) else float("nan")

    turn = float("nan")
    turn_se = float("nan")
    if abs(b2) > 1e-12:
        turn = -b1 / (2.0 * b2)
        V = vcov[:2, :2]
        d_db1 = -1.0 / (2.0 * b2)
        d_db2 = b1 / (2.0 * (b2 ** 2))
        g = np.array([d_db1, d_db2], dtype=float)
        var_t = float(g.T @ V @ g)
        if var_t > 0 and np.isfinite(var_t):
            turn_se = math.sqrt(var_t)

    f1, fp1 = first_stage_joint_f(x_dm[:, 0], z_dm, w_dm, firm_ids)
    f2, fp2 = first_stage_joint_f(x_dm[:, 1], z_dm, w_dm, firm_ids)

    xraw = x_endog[:, 0]
    q01 = float(np.quantile(xraw, 0.01))
    q99 = float(np.quantile(xraw, 0.99))

    return {
        "dep_var": dep_var,
        "nobs": int(nobs),
        "n_firms": int(firm_ids.max()) + 1 if nobs else 0,
        "n_dates": int(time_ids.max()) + 1 if nobs else 0,
        "beta_ind_own": b1,
        "se_ind_own": s1,
        "t_ind_own": t1,
        "p_ind_own": p1,
        "beta_ind_own_sq": b2,
        "se_ind_own_sq": s2,
        "t_ind_own_sq": t2,
        "p_ind_own_sq": p2,
        "turning_point": turn,
        "turning_point_se": turn_se,
        "turning_in_q01_q99": bool((turn >= q01) and (turn <= q99)) if np.isfinite(turn) else False,
        "ind_own_q01": q01,
        "ind_own_q99": q99,
        "first_stage_jointF_ind_own": f1,
        "first_stage_jointP_ind_own": fp1,
        "first_stage_jointF_ind_own_sq": f2,
        "first_stage_jointP_ind_own_sq": fp2,
        "cutoff": CUTOFF,
        "bandwidth": BANDWIDTH,
        "excluded_instruments": ";".join([f"z_cutoff_x_running_pow{k}" for k in range(POLY_DEG + 1)]),
        "controls": "c_firm_size;c_dollar_vol;" + ";".join([f"running_pow{k}" for k in range(1, POLY_DEG + 1)]),
    }


def write_summary(rows):
    lines = []
    lines.append("# Nonlinear IV (Quadratic) Results")
    lines.append("")
    lines.append("Specification:")
    lines.append("- 2SLS with endogenous: `ind_own`, `ind_own^2`")
    lines.append(f"- Excluded instruments: `z*running^k, k=0..{POLY_DEG}`")
    lines.append(f"- Controls: `c_firm_size`, `c_dollar_vol`, `running^k, k=1..{POLY_DEG}`")
    lines.append("- FE: firm + date (via two-way demeaning)")
    lines.append(f"- Local sample: cutoff={CUTOFF:.0f}, bandwidth={BANDWIDTH:.0f}")
    lines.append("")
    for r in rows:
        lines.append(f"## {r['dep_var']}")
        lines.append(
            f"- b1(ind_own)={r['beta_ind_own']:.4g} (p={r['p_ind_own']:.3g}), "
            f"b2(ind_own^2)={r['beta_ind_own_sq']:.4g} (p={r['p_ind_own_sq']:.3g})"
        )
        lines.append(
            f"- turning point={r['turning_point']:.4f}, in_support={r['turning_in_q01_q99']} "
            f"[q01={r['ind_own_q01']:.4f}, q99={r['ind_own_q99']:.4f}]"
        )
        lines.append(
            f"- first-stage joint F: ind_own={r['first_stage_jointF_ind_own']:.3f} "
            f"(p={r['first_stage_jointP_ind_own']:.3g}); "
            f"ind_own^2={r['first_stage_jointF_ind_own_sq']:.3f} "
            f"(p={r['first_stage_jointP_ind_own_sq']:.3g})"
        )
        lines.append("")
    OUT_SUMMARY.write_text("\n".join(lines), encoding="utf-8")


def main():
    rank_map = load_rank_map()
    rows = [run_one(dep, rank_map) for dep in DEP_VARS]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with OUT_RESULTS.open("w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)
    write_summary(rows)

    print(f"Saved: {OUT_RESULTS}")
    print(f"Saved: {OUT_SUMMARY}")


if __name__ == "__main__":
    main()
