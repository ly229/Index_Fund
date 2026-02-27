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
WINDOW_QUARTERS = 16  # rolling 4-year windows

RECESSION_WINDOWS = [("2001Q1", "2001Q4"), ("2007Q4", "2009Q2"), ("2019Q4", "2020Q2")]

OUT_CSV = OUT_DIR / "beta_over_time_rolling_window.csv"


def prepare_base(df: pd.DataFrame, dep_var: str) -> pd.DataFrame:
    d = df[["cusip", "date", dep_var] + RHS_VARS].copy()
    d["date"] = pd.to_datetime(d["date"])
    d["quarter"] = d["date"].dt.to_period("Q")
    for c in [dep_var] + RHS_VARS:
        d[c] = pd.to_numeric(d[c], errors="coerce")
        d.loc[~np.isfinite(d[c]), c] = np.nan
    d = d.dropna(subset=[dep_var] + RHS_VARS)
    return d


def run_window_reg(d: pd.DataFrame, dep_var: str, q_start: pd.Period, q_end: pd.Period) -> tuple[float, float, float, int, int, int]:
    w = d[(d["quarter"] >= q_start) & (d["quarter"] <= q_end)].copy()
    w = w.set_index(["cusip", "date"]).sort_index()
    model = PanelOLS.from_formula(
        f"{dep_var} ~ 1 + {' + '.join(RHS_VARS)} + EntityEffects + TimeEffects",
        data=w,
        drop_absorbed=True,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        fit = model.fit(cov_type="clustered", cluster_entity=True, cluster_time=True)
        beta = float(fit.params["ind_own"])
        se = float(fit.std_errors["ind_own"])
        pval = float(fit.pvalues["ind_own"])
    nobs = int(fit.nobs)
    n_entities = int(w.index.get_level_values(0).nunique())
    n_time = int(w.index.get_level_values(1).nunique())
    return beta, se, pval, nobs, n_entities, n_time


def estimate_betas(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dep in DEP_VARS:
        base = prepare_base(df, dep)
        quarters = np.array(sorted(base["quarter"].unique()))
        for i in range(WINDOW_QUARTERS - 1, len(quarters)):
            q_end = quarters[i]
            q_start = quarters[i - WINDOW_QUARTERS + 1]
            try:
                beta, se, pval, nobs, n_entities, n_time = run_window_reg(base, dep, q_start, q_end)
                rows.append(
                    {
                        "dep_var": dep,
                        "window_start_q": str(q_start),
                        "window_end_q": str(q_end),
                        "window_end_date": q_end.to_timestamp(how="end"),
                        "beta_ind_own": beta,
                        "se_ind_own": se,
                        "pval_ind_own": pval,
                        "ci95_lo": beta - 1.96 * se,
                        "ci95_hi": beta + 1.96 * se,
                        "nobs": nobs,
                        "entities": n_entities,
                        "time_periods": n_time,
                    }
                )
            except Exception:
                continue
    out = pd.DataFrame(rows).sort_values(["dep_var", "window_end_date"]).reset_index(drop=True)
    return out


def add_recession_shading(ax: plt.Axes) -> None:
    for qs, qe in RECESSION_WINDOWS:
        s = pd.Period(qs, freq="Q").to_timestamp(how="start")
        e = pd.Period(qe, freq="Q").to_timestamp(how="end")
        ax.axvspan(s, e, color="#add8e6", alpha=0.35, lw=0)


def plot_by_dep(res: pd.DataFrame) -> None:
    labels = {
        "amihud_illiq": "Amihud illiquidity",
        "volatility": "Volatility",
        "price_info": "Price informativeness",
    }
    colors = {
        "amihud_illiq": "#1f77b4",
        "volatility": "#d62728",
        "price_info": "#2ca02c",
    }
    for dep in DEP_VARS:
        d = res[res["dep_var"] == dep].copy()
        x = pd.to_datetime(d["window_end_date"])
        y = d["beta_ind_own"].to_numpy()
        lo = d["ci95_lo"].to_numpy()
        hi = d["ci95_hi"].to_numpy()

        fig, ax = plt.subplots(figsize=(12.8, 4.8))
        add_recession_shading(ax)
        ax.fill_between(x, lo, hi, color=colors[dep], alpha=0.18, label="95% CI")
        ax.plot(x, y, color=colors[dep], linewidth=2.0, label="Beta")
        ax.axhline(0, color="#666666", linestyle="--", linewidth=1)
        ax.set_title(f"Time-Varying Beta of ind_own: {labels[dep]} (Rolling {WINDOW_QUARTERS}Q Panel FE)")
        ax.set_xlabel("Date (Year-Quarter)")
        ax.set_ylabel("Coefficient")
        ax.grid(axis="y", alpha=0.25, linestyle=":")

        ticks = pd.period_range(x.min().to_period("Q"), x.max().to_period("Q"), freq="8Q")
        ax.set_xticks(ticks.to_timestamp(how="end"))
        ax.set_xticklabels([str(t) for t in ticks], rotation=45, ha="right", fontsize=8)
        ax.legend(frameon=False, loc="upper left")
        fig.tight_layout()
        out_plot = OUT_DIR / f"beta_over_time_{dep}.svg"
        fig.savefig(out_plot, format="svg", dpi=220, bbox_inches="tight")
        plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA_PATH)
    res = estimate_betas(df)
    res.to_csv(OUT_CSV, index=False)
    plot_by_dep(res)
    print(f"Saved: {OUT_CSV}")
    for dep in DEP_VARS:
        print(f"Saved: {OUT_DIR / f'beta_over_time_{dep}.svg'}")


if __name__ == "__main__":
    main()
