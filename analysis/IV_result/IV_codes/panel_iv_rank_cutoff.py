#!/usr/bin/env python3
import csv
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np


MAIN_PATH = Path("data/df_regression.csv")
RANK_PATH = Path("data/Russell_3_rank.csv")
OUT_PATH = Path("analysis/iv_panel_results_rank_cutoff.csv")

DEP_VARS = ["amihud_illiq", "volatility", "price_info"]
ENDOG = "ind_own"
CONTROLS = ["mktcap", "c_dollar_vol", "r_tno"]
CUTOFF = 1000.0
BANDWIDTH = 150.0
RUNNING_VAR = "rank_minus_cutoff"
INSTR = "z_rank_le_cutoff"


@dataclass
class FitResult:
    dep_var: str
    nobs: int
    n_firms: int
    n_dates: int
    beta_endog: float
    se_endog: float
    t_endog: float
    p_endog: float
    first_stage_f: float
    first_stage_p: float
    n_clusters: int


def normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def parse_float(x: str):
    if x is None:
        return None
    x = x.strip()
    if x == "":
        return None
    try:
        v = float(x)
        if not math.isfinite(v):
            return None
        return v
    except ValueError:
        return None


def parse_date(x: str):
    return datetime.strptime(x, "%Y-%m-%d")


def rank_year_for_panel_date(dt: datetime) -> int:
    return dt.year if dt.month >= 6 else dt.year - 1


def safe_log1p(v: float):
    if v is None:
        return None
    if v <= -1.0 or (not math.isfinite(v)):
        return None
    out = math.log1p(v)
    if not math.isfinite(out):
        return None
    return out


def load_rank_cutoff_map(path: Path):
    rank_map = {}
    dup_count = 0
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cusip = row.get("cusip", "").strip()
            d = row.get("date", "").strip()
            r = parse_float(row.get("Rank", ""))
            if not cusip or not d or r is None:
                continue
            dt = parse_date(d)
            key = (cusip, dt.year)
            if key in rank_map:
                dup_count += 1
                if r < rank_map[key]:
                    rank_map[key] = r
            else:
                rank_map[key] = r
    return rank_map, dup_count


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

        delta = np.max(np.abs(x - prev))
        if delta < tol:
            break

    return x


def ols_fit(X: np.ndarray, y: np.ndarray):
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    ssr = float(resid.T @ resid)
    return beta, resid, ssr


def f_test_partial(x: np.ndarray, z: np.ndarray, w: np.ndarray):
    n = x.shape[0]
    Xr = w
    Xu = np.column_stack([z, w])

    _, _, ssr_r = ols_fit(Xr, x)
    _, _, ssr_u = ols_fit(Xu, x)

    q = 1
    k_u = Xu.shape[1]
    denom_df = n - k_u
    if denom_df <= 0 or ssr_u <= 0:
        return float("nan"), float("nan")

    fstat = ((ssr_r - ssr_u) / q) / (ssr_u / denom_df)
    # Large-sample approximation for p-value with q=1: F(1, large) ~ Chi-square(1)
    pval = 2.0 * (1.0 - normal_cdf(math.sqrt(max(fstat, 0.0))))
    return float(fstat), float(max(min(pval, 1.0), 0.0))


def ols_clustered_coef_se(y: np.ndarray, X: np.ndarray, cluster_ids: np.ndarray):
    n, k = X.shape
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    u = y - X @ beta

    XtX = X.T @ X
    XtX_inv = np.linalg.pinv(XtX)

    n_clusters = int(cluster_ids.max()) + 1
    meat = np.zeros((k, k), dtype=float)
    for g in range(n_clusters):
        idx = cluster_ids == g
        if not np.any(idx):
            continue
        Xg = X[idx, :]
        ug = u[idx]
        s = Xg.T @ ug
        meat += np.outer(s, s)

    # Small-sample finite-cluster correction.
    if n_clusters > 1 and (n - k) > 0:
        corr = (n_clusters / (n_clusters - 1.0)) * ((n - 1.0) / (n - k))
    else:
        corr = 1.0
    vcov = corr * (XtX_inv @ meat @ XtX_inv)
    se = np.sqrt(np.maximum(np.diag(vcov), 0.0))
    return beta, se


def first_stage_cluster_f(x: np.ndarray, z: np.ndarray, w: np.ndarray, cluster_ids: np.ndarray):
    X = np.column_stack([z, w])
    beta, se = ols_clustered_coef_se(x, X, cluster_ids)
    b = float(beta[0])
    s = float(se[0])
    if s <= 0 or (not np.isfinite(s)):
        return float("nan"), float("nan")
    t = b / s
    f = t * t
    p = 2.0 * (1.0 - normal_cdf(abs(t)))
    return float(f), float(max(min(p, 1.0), 0.0))


def iv_2sls_clustered(y: np.ndarray, x_endog: np.ndarray, z_excl: np.ndarray, w: np.ndarray, cluster_ids: np.ndarray):
    X = np.column_stack([x_endog, w])
    Z = np.column_stack([z_excl, w])

    n = X.shape[0]
    k = X.shape[1]

    Qxz = (X.T @ Z) / n
    Qzz = (Z.T @ Z) / n
    Qzz_inv = np.linalg.pinv(Qzz)

    middle = Qxz @ Qzz_inv @ Qxz.T
    middle_inv = np.linalg.pinv(middle)

    beta = middle_inv @ (Qxz @ Qzz_inv @ ((Z.T @ y) / n))

    u = y - X @ beta
    # Cluster-robust instrument moment covariance:
    # S = (1/n) * sum_g G_g G_g', where G_g = sum_{i in g} z_i * u_i
    n_clusters = int(cluster_ids.max()) + 1
    l = Z.shape[1]
    S = np.zeros((l, l), dtype=float)
    for g in range(n_clusters):
        idx = cluster_ids == g
        if not np.any(idx):
            continue
        Zg = Z[idx, :]
        ug = u[idx]
        Gg = Zg.T @ ug
        S += np.outer(Gg, Gg)
    S /= n

    meat = Qxz @ Qzz_inv @ S @ Qzz_inv @ Qxz.T
    vcov = middle_inv @ meat @ middle_inv / n

    # Small-sample finite-cluster correction.
    if n_clusters > 1 and (n - k) > 0:
        corr = (n_clusters / (n_clusters - 1.0)) * ((n - 1.0) / (n - k))
    else:
        corr = 1.0
    vcov *= corr

    se = np.sqrt(np.maximum(np.diag(vcov), 0.0))

    return beta, se


def prepare_dataset(dep_var: str, rank_map: dict, cutoff: float, bandwidth):
    y_vals = []
    x_vals = []
    z_vals = []
    w_vals = []
    firm_ids = []
    time_ids = []

    firm_to_id = {}
    time_to_id = {}

    with MAIN_PATH.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cusip = row.get("cusip", "").strip()
            d = row.get("date", "").strip()
            if not cusip or not d:
                continue

            y = parse_float(row.get(dep_var, ""))
            x = parse_float(row.get(ENDOG, ""))
            c1 = parse_float(row.get(CONTROLS[0], ""))
            c2 = parse_float(row.get(CONTROLS[1], ""))
            c3 = parse_float(row.get(CONTROLS[2], ""))
            c1 = safe_log1p(c1)
            c2 = safe_log1p(c2)
            if any(v is None for v in [y, x, c1, c2, c3]):
                continue

            dt = parse_date(d)
            ryear = rank_year_for_panel_date(dt)
            rank = rank_map.get((cusip, ryear), None)
            if rank is None:
                continue
            running = rank - cutoff
            if bandwidth is not None and abs(running) > bandwidth:
                continue
            z = 1.0 if rank <= cutoff else 0.0

            if cusip not in firm_to_id:
                firm_to_id[cusip] = len(firm_to_id)
            if d not in time_to_id:
                time_to_id[d] = len(time_to_id)

            y_vals.append(y)
            x_vals.append(x)
            z_vals.append(z)
            w_vals.append((c1, c2, c3, running))
            firm_ids.append(firm_to_id[cusip])
            time_ids.append(time_to_id[d])

    y = np.asarray(y_vals, dtype=float)
    x = np.asarray(x_vals, dtype=float)
    z = np.asarray(z_vals, dtype=float)
    w = np.asarray(w_vals, dtype=float)
    firm_ids = np.asarray(firm_ids, dtype=int)
    time_ids = np.asarray(time_ids, dtype=int)

    finite_mask = (
        np.isfinite(y)
        & np.isfinite(x)
        & np.isfinite(z)
        & np.all(np.isfinite(w), axis=1)
    )
    y = y[finite_mask]
    x = x[finite_mask]
    z = z[finite_mask]
    w = w[finite_mask]
    firm_ids = firm_ids[finite_mask]
    time_ids = time_ids[finite_mask]

    return y, x, z, w, firm_ids, time_ids


def run_one(dep_var: str, rank_map: dict, cutoff: float, bandwidth):
    y, x, z, w, firm_ids, time_ids = prepare_dataset(dep_var, rank_map, cutoff, bandwidth)
    nobs = y.shape[0]

    M = np.column_stack([y, x, z, w])
    M_dm = two_way_demean(M, firm_ids, time_ids)

    y_dm = M_dm[:, 0]
    x_dm = M_dm[:, 1]
    z_dm = M_dm[:, 2]
    w_dm = M_dm[:, 3:]

    beta, se = iv_2sls_clustered(y_dm, x_dm, z_dm, w_dm, firm_ids)

    b = float(beta[0])
    s = float(se[0])
    t = b / s if s > 0 else float("nan")
    p = 2.0 * (1.0 - normal_cdf(abs(t))) if np.isfinite(t) else float("nan")

    fstat, fp = first_stage_cluster_f(x_dm, z_dm, w_dm, firm_ids)

    return FitResult(
        dep_var=dep_var,
        nobs=nobs,
        n_firms=int(firm_ids.max()) + 1 if nobs else 0,
        n_dates=int(time_ids.max()) + 1 if nobs else 0,
        beta_endog=b,
        se_endog=s,
        t_endog=t,
        p_endog=p,
        first_stage_f=fstat,
        first_stage_p=fp,
        n_clusters=int(firm_ids.max()) + 1 if nobs else 0,
    )


def main():
    rank_map, dup_count = load_rank_cutoff_map(RANK_PATH)
    print(f"Loaded rank map: {len(rank_map)} cusip-year keys, duplicates resolved: {dup_count}")

    results = []
    for dep in DEP_VARS:
        print(f"Running IV for {dep} ...")
        res = run_one(dep, rank_map, CUTOFF, BANDWIDTH)
        results.append(res)
        print(
            f"{dep}: n={res.nobs}, firms={res.n_firms}, dates={res.n_dates}, "
            f"beta(ind_own)={res.beta_endog:.6g}, se={res.se_endog:.6g}, "
            f"t={res.t_endog:.4f}, p={res.p_endog:.4g}, F1st={res.first_stage_f:.4f}, "
            f"clusters={res.n_clusters}"
        )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "dep_var",
            "nobs",
            "n_firms",
            "n_dates",
            "endog_var",
            "instrument",
            "controls",
            "beta_endog",
            "se_endog",
            "t_endog",
            "p_endog",
            "first_stage_f",
            "first_stage_p",
            "n_clusters",
            "fixed_effects",
            "se_type",
            "bandwidth",
            "cutoff",
        ])
        for r in results:
            writer.writerow([
                r.dep_var,
                r.nobs,
                r.n_firms,
                r.n_dates,
                ENDOG,
                INSTR,
                ";".join(CONTROLS + [RUNNING_VAR]),
                r.beta_endog,
                r.se_endog,
                r.t_endog,
                r.p_endog,
                r.first_stage_f,
                r.first_stage_p,
                r.n_clusters,
                "cusip_fe;date_fe",
                "firm_clustered",
                BANDWIDTH,
                CUTOFF,
            ])

    print(f"Saved: {OUT_PATH}")


if __name__ == "__main__":
    main()
