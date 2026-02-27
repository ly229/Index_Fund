#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path
import importlib.util

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SCRIPT_DIR = Path(__file__).resolve().parent
module_path = None
for cand in [SCRIPT_DIR / "iv_quadratic_rank_cutoff.py", SCRIPT_DIR / "IV" / "iv_quadratic_rank_cutoff.py"]:
    if cand.exists():
        module_path = cand
        break
if module_path is None:
    raise FileNotFoundError("Cannot find iv_quadratic_rank_cutoff.py")

spec = importlib.util.spec_from_file_location("iv_quadratic_rank_cutoff", module_path)
base = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(base)


OUT_DIR = Path("analysis/nonlinearity")
OUT_FIRST_STAGE = OUT_DIR / "iv_first_stage_cutoff2000_global.svg"
OUT_FITTED_ALL = OUT_DIR / "iv_fitted_indownhat_vs_depvars_cutoff2000_global.svg"

DEP_VARS = ["amihud_illiq", "volatility", "price_info"]
DEP_LABELS = {
    "amihud_illiq": "Amihud illiquidity",
    "volatility": "Volatility",
    "price_info": "Price informativeness",
}
COLORS = {"amihud_illiq": "#1f77b4", "volatility": "#d62728", "price_info": "#2ca02c"}


def binned_xy(x: np.ndarray, y: np.ndarray, n_bins: int = 35):
    q = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.quantile(x, q)
    edges[0] -= 1e-12
    edges[-1] += 1e-12
    xs, ys = [], []
    for i in range(n_bins):
        m = (x > edges[i]) & (x <= edges[i + 1])
        if np.sum(m) < 20:
            continue
        xs.append(float(np.mean(x[m])))
        ys.append(float(np.mean(y[m])))
    return np.asarray(xs), np.asarray(ys)


def plot_first_stage(rank_map):
    # Build sample with only first-stage ingredients.
    y, x_endog, z_excl, w, firm_ids, time_ids = base.prepare("amihud_illiq", rank_map)
    x = x_endog[:, 0]
    z0 = z_excl[:, 0]  # main cutoff indicator instrument

    M = np.column_stack([x, z0, w])
    M_dm = base.two_way_demean(M, firm_ids, time_ids)
    x_tilde = M_dm[:, 0]
    z_tilde = M_dm[:, 1]

    X = np.column_stack([np.ones_like(z_tilde), z_tilde])
    b, *_ = np.linalg.lstsq(X, x_tilde, rcond=None)
    xg = np.linspace(np.min(z_tilde), np.max(z_tilde), 300)
    yg = b[0] + b[1] * xg

    bx, by = binned_xy(z_tilde, x_tilde, n_bins=40)

    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    ax.scatter(bx, by, s=16, color="#4c78a8", alpha=0.9, label="Binned means")
    ax.plot(xg, yg, color="#e45756", linewidth=2.0, label="Linear fit")
    ax.axhline(0, color="#666", linestyle="--", linewidth=1)
    ax.axvline(0, color="#666", linestyle=":", linewidth=1)
    ax.set_title("IV First Stage (FE-residualized): ind_own on cutoff instrument")
    ax.set_xlabel("Residualized instrument z_cutoff")
    ax.set_ylabel("Residualized ind_own")
    ax.grid(axis="y", alpha=0.25, linestyle=":")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT_FIRST_STAGE, format="svg", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_fitted_lines(rank_map):
    fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.8), sharex=False)

    for ax, dep in zip(axes, DEP_VARS):
        y, x_endog, z_excl, w, firm_ids, time_ids = base.prepare(dep, rank_map)
        x = x_endog[:, 0]

        M = np.column_stack([y, x, z_excl, w])
        M_dm = base.two_way_demean(M, firm_ids, time_ids)
        y_dm = M_dm[:, 0]
        x_dm = M_dm[:, 1]
        q = z_excl.shape[1]
        z_dm = M_dm[:, 2 : 2 + q]
        w_dm = M_dm[:, 2 + q :]

        # First-stage fitted endogenous variable.
        Xfs = np.column_stack([z_dm, w_dm])
        b_fs, *_ = np.linalg.lstsq(Xfs, x_dm, rcond=None)
        xhat_dm = Xfs @ b_fs

        # Nonlinear fitted curve between y_dm and xhat_dm (quadratic).
        mask = np.isfinite(xhat_dm) & np.isfinite(y_dm)
        xhat_dm = xhat_dm[mask]
        y_dm = y_dm[mask]
        X2 = np.column_stack([np.ones_like(xhat_dm), xhat_dm, xhat_dm * xhat_dm])
        b2, *_ = np.linalg.lstsq(X2, y_dm, rcond=None)
        xg = np.linspace(np.quantile(xhat_dm, 0.01), np.quantile(xhat_dm, 0.99), 300)
        yg = b2[0] + b2[1] * xg + b2[2] * xg * xg

        bx, by = binned_xy(xhat_dm, y_dm, n_bins=35)
        ax.scatter(bx, by, s=16, color=COLORS[dep], alpha=0.85)
        ax.plot(xg, yg, color="#222222", linewidth=2.0)
        ax.axhline(0, color="#666", linestyle="--", linewidth=1)
        ax.axvline(0, color="#666", linestyle=":", linewidth=1)
        ax.set_title(DEP_LABELS[dep], fontsize=10.5)
        ax.set_xlabel("Residualized first-stage fitted ind_own")
        ax.grid(axis="y", alpha=0.25, linestyle=":")

    axes[0].set_ylabel("Residualized dependent variable")
    fig.suptitle("IV Nonlinear Fitted Curve: ind_own_hat vs Dependent Variables (Cutoff=2000, Global)", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT_FITTED_ALL, format="svg", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    base.CUTOFF = 2000.0
    base.BANDWIDTH = 1e9
    base.POLY_DEG = 2

    rank_map = base.load_rank_map()
    plot_first_stage(rank_map)
    plot_fitted_lines(rank_map)
    print(f"Saved: {OUT_FIRST_STAGE}")
    print(f"Saved: {OUT_FITTED_ALL}")


if __name__ == "__main__":
    main()
