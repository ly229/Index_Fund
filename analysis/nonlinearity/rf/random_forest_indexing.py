#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DATA_PATH = Path("data/df_regression.csv")
OUT_DIR = Path("analysis/nonlinearity/rf")

DEP_VARS = ["amihud_illiq", "volatility", "price_info"]
MAIN = "ind_own"
CTRL_VARS = ["c_firm_size", "c_dollar_vol"]
DATE_COL = "date"

MAX_TRAIN_PER_DATE = 1500
MAX_EVAL_PER_DATE = 200
N_ESTIMATORS = 120
MAX_DEPTH = 14
MIN_SAMPLES_LEAF = 100
MAX_FEATURES = 0.8
MAX_SAMPLES = 0.55
RANDOM_STATE = 42
GRID_SIZE = 220
BATCH_SIZE = 25
EVAL_COLUMNS = [MAIN] + CTRL_VARS

OUT_CURVE = OUT_DIR / "rf_indexing_curve_grid.csv"
OUT_MODEL_SUMMARY = OUT_DIR / "rf_indexing_model_summary.csv"
OUT_SUMMARY = OUT_DIR / "rf_indexing_summary.md"
OUT_PLOT_SCORE = OUT_DIR / "rf_indexing_score.svg"
OUT_PLOT_CURVES = OUT_DIR / "rf_relationship_all3.svg"


def complete_case_sample(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["cusip", DATE_COL, MAIN] + CTRL_VARS + DEP_VARS
    d = df[cols].copy()
    d[DATE_COL] = pd.to_datetime(d[DATE_COL])
    for c in [MAIN] + CTRL_VARS + DEP_VARS:
        d[c] = pd.to_numeric(d[c], errors="coerce")
        d.loc[~np.isfinite(d[c]), c] = np.nan
    d = d.dropna(subset=[MAIN] + CTRL_VARS + DEP_VARS)
    return d


def sample_by_date(df: pd.DataFrame, max_per_date: int, random_state: int) -> pd.DataFrame:
    sampled = (
        df.groupby(DATE_COL, group_keys=False)
        .apply(lambda g: g.sample(n=min(len(g), max_per_date), random_state=random_state))
        .reset_index(drop=True)
    )
    return sampled.sort_values([DATE_COL, "cusip"]).reset_index(drop=True)


def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    base = df[[MAIN] + CTRL_VARS].copy().reset_index(drop=True)
    dates = pd.get_dummies(df[DATE_COL].dt.to_period("Q").astype(str), prefix="date", dtype=np.int8)
    X = pd.concat([base, dates.reset_index(drop=True)], axis=1)
    feature_cols = X.columns.tolist()
    return X, feature_cols


def fit_rf(X: pd.DataFrame, y: pd.Series) -> RandomForestRegressor:
    model = RandomForestRegressor(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        max_features=MAX_FEATURES,
        bootstrap=True,
        oob_score=True,
        max_samples=MAX_SAMPLES,
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        model.fit(X, y)
    return model


def predict_partial_dependence(
    model: RandomForestRegressor,
    X_eval: pd.DataFrame,
    grid: np.ndarray,
    main_col: str = MAIN,
    batch_size: int = BATCH_SIZE,
) -> np.ndarray:
    X_base = X_eval.copy()
    vals = []
    for start in range(0, len(grid), batch_size):
        stop = min(start + batch_size, len(grid))
        batch = grid[start:stop]
        rows = []
        for x in batch:
            X_tmp = X_base.copy()
            X_tmp[main_col] = x
            rows.append(model.predict(X_tmp).mean())
        vals.extend(rows)
    return np.asarray(vals, dtype=float)


def summarize_support(x: pd.Series) -> dict[str, float]:
    return {
        "min": float(x.min()),
        "q01": float(x.quantile(0.01)),
        "q05": float(x.quantile(0.05)),
        "q50": float(x.quantile(0.50)),
        "q95": float(x.quantile(0.95)),
        "q99": float(x.quantile(0.99)),
        "max": float(x.max()),
    }


def fit_one(df: pd.DataFrame, dep_var: str) -> tuple[dict, pd.DataFrame]:
    d = df[["cusip", DATE_COL, MAIN] + CTRL_VARS + [dep_var]].copy()
    d = d.dropna(subset=[MAIN] + CTRL_VARS + [dep_var]).reset_index(drop=True)

    train = sample_by_date(d, MAX_TRAIN_PER_DATE, RANDOM_STATE)
    eval_df = sample_by_date(d, MAX_EVAL_PER_DATE, RANDOM_STATE + 7)

    X_train, feature_cols = build_features(train)
    X_eval, _ = build_features(eval_df)
    y_train = train[dep_var].astype(float).reset_index(drop=True)
    y_eval = eval_df[dep_var].astype(float).reset_index(drop=True)

    model = fit_rf(X_train, y_train)
    pred_train = model.predict(X_train)
    pred_eval = model.predict(X_eval)

    support = summarize_support(train[MAIN])
    xgrid = np.linspace(support["q01"], support["q99"], GRID_SIZE)
    x_ref = support["q50"]
    pd_curve = predict_partial_dependence(model, X_eval, xgrid)
    pd_ref = float(predict_partial_dependence(model, X_eval, np.asarray([x_ref]))[0])
    delta = pd_curve - pd_ref

    row = {
        "dep_var": dep_var,
        "train_nobs": int(len(train)),
        "eval_nobs": int(len(eval_df)),
        "entities_train": int(train["cusip"].nunique()),
        "dates_train": int(train[DATE_COL].nunique()),
        "entities_eval": int(eval_df["cusip"].nunique()),
        "dates_eval": int(eval_df[DATE_COL].nunique()),
        "oob_r2": float(model.oob_score_),
        "train_r2": float(r2_score(y_train, pred_train)),
        "eval_r2": float(r2_score(y_eval, pred_eval)),
        "ind_own_min": support["min"],
        "ind_own_q01": support["q01"],
        "ind_own_q05": support["q05"],
        "ind_own_q50": support["q50"],
        "ind_own_q95": support["q95"],
        "ind_own_q99": support["q99"],
        "ind_own_max": support["max"],
    }

    out = pd.DataFrame(
        {
            "dep_var": dep_var,
            "ind_own": xgrid,
            "pred_level": pd_curve,
            "delta_from_median": delta,
        }
    )
    return row, out


def zscore(v: np.ndarray) -> np.ndarray:
    s = float(np.std(v))
    if not np.isfinite(s) or s <= 1e-12:
        return np.zeros_like(v, dtype=float)
    return (v - float(np.mean(v))) / s


def build_score_and_plots(model_df: pd.DataFrame, curve_df: pd.DataFrame) -> pd.DataFrame:
    x_lo = float(curve_df["ind_own"].min())
    x_hi = float(curve_df["ind_own"].max())
    xgrid = np.sort(curve_df["ind_own"].unique())
    if len(xgrid) != GRID_SIZE:
        xgrid = np.linspace(x_lo, x_hi, GRID_SIZE)

    curves = {}
    for dep in DEP_VARS:
        curves[dep] = (
            curve_df.loc[curve_df["dep_var"] == dep, ["ind_own", "delta_from_median"]]
            .sort_values("ind_own")["delta_from_median"]
            .to_numpy(dtype=float)
        )

    score = -zscore(curves["amihud_illiq"]) - zscore(curves["volatility"]) + zscore(curves["price_info"])
    best_idx = int(np.argmax(score))
    best_x = float(xgrid[best_idx])
    score_max = float(score[best_idx])
    score_q95 = float(np.quantile(score, 0.95))
    top_mask = score >= score_q95
    good_x = xgrid[top_mask]
    good_lo = float(good_x.min()) if len(good_x) else np.nan
    good_hi = float(good_x.max()) if len(good_x) else np.nan

    out = pd.DataFrame(
        {
            "ind_own": xgrid,
            "score_total": score,
            "delta_amihud": curves["amihud_illiq"],
            "delta_volatility": curves["volatility"],
            "delta_price_info": curves["price_info"],
            "is_top5pct_score": top_mask.astype(int),
        }
    )

    out = pd.concat(
        [
            out,
            pd.DataFrame(
                [
                    {
                        "ind_own": best_x,
                        "score_total": score_max,
                        "delta_amihud": np.nan,
                        "delta_volatility": np.nan,
                        "delta_price_info": np.nan,
                        "is_top5pct_score": 2,
                    },
                    {
                        "ind_own": good_lo,
                        "score_total": score_q95,
                        "delta_amihud": np.nan,
                        "delta_volatility": np.nan,
                        "delta_price_info": np.nan,
                        "is_top5pct_score": 3,
                    },
                    {
                        "ind_own": good_hi,
                        "score_total": score_q95,
                        "delta_amihud": np.nan,
                        "delta_volatility": np.nan,
                        "delta_price_info": np.nan,
                        "is_top5pct_score": 4,
                    },
                ]
            ),
        ],
        ignore_index=True,
    )

    fig, ax = plt.subplots(figsize=(10.8, 4.8))
    ax.plot(xgrid, score, color="#1f77b4", linewidth=2.1, label="Market quality score")
    ax.axvline(best_x, color="#d62728", linestyle="--", linewidth=1.4, label=f"Best indexing ~ {best_x:.3f}")
    if np.isfinite(good_lo) and np.isfinite(good_hi):
        ax.axvspan(good_lo, good_hi, color="#9ecae1", alpha=0.32, lw=0, label="Top 5% score range")
    ax.set_title("How Much Indexing Is Good? Random Forest Composite Score")
    ax.set_xlabel("ind_own")
    ax.set_ylabel("Standardized composite score")
    ax.grid(axis="y", alpha=0.25, linestyle=":")
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(OUT_PLOT_SCORE, format="svg", dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(14.8, 4.4), sharex=True)
    meta = {
        "amihud_illiq": ("Illiquidity", "#8c2d04", "Lower is better"),
        "volatility": ("Volatility", "#08519c", "Lower is better"),
        "price_info": ("Price Informativeness", "#006d2c", "Higher is better"),
    }
    for ax, dep in zip(axes, DEP_VARS):
        y = curves[dep]
        ax.plot(xgrid, y, color=meta[dep][1], linewidth=2.0)
        ax.axhline(0.0, color="0.35", linewidth=0.9, linestyle=":")
        ax.axvline(best_x, color="#d62728", linewidth=1.0, linestyle="--")
        ax.set_title(meta[dep][0], fontsize=11)
        ax.set_xlabel("ind_own")
        ax.grid(axis="y", alpha=0.22, linestyle=":")
        ax.text(0.03, 0.92, meta[dep][2], transform=ax.transAxes, fontsize=9, va="top")
    axes[0].set_ylabel("Partial dependence change from median")
    fig.suptitle("Random Forest Nonlinearity Across Market-Quality Measures", fontsize=14, y=0.98)
    fig.tight_layout()
    fig.savefig(OUT_PLOT_CURVES, format="svg", dpi=220, bbox_inches="tight")
    plt.close(fig)

    summary_row = {
        "best_ind_own": best_x,
        "best_score": score_max,
        "top5pct_lo": good_lo,
        "top5pct_hi": good_hi,
        "grid_q95_score": score_q95,
    }
    return out, summary_row


def write_summary(model_df: pd.DataFrame, score_row: dict) -> None:
    lines: list[str] = []
    lines.append("# Random Forest Nonlinearity: How Much Indexing Is Good?")
    lines.append("")
    lines.append("Specification:")
    lines.append("- Random forest regressor with date dummies and the baseline controls `c_firm_size` and `c_dollar_vol`.")
    lines.append("- Outcome models estimated on the common complete-case sample.")
    lines.append("- Partial dependence is computed over an `ind_own` grid, holding the remaining features at their empirical distribution.")
    lines.append("")
    lines.append("## Sample")
    for _, r in model_df.iterrows():
        lines.append(
            f"- `{r['dep_var']}`: train n={int(r['train_nobs'])}, eval n={int(r['eval_nobs'])}, "
            f"oob R2={r['oob_r2']:.3f}, train R2={r['train_r2']:.3f}, eval R2={r['eval_r2']:.3f}"
        )
    lines.append("")
    lines.append("## Composite Answer")
    lines.append(f"- Composite optimum indexing level: `ind_own ≈ {score_row['best_ind_own']:.4f}`")
    lines.append(f"- High-quality range (top 5% score): `[{score_row['top5pct_lo']:.4f}, {score_row['top5pct_hi']:.4f}]`")
    lines.append("")
    lines.append("## Outcome Curves")
    for dep in DEP_VARS:
        r = model_df.loc[model_df["dep_var"] == dep].iloc[0]
        lines.append(
            f"- `{dep}`: oob R2={r['oob_r2']:.3f}, eval R2={r['eval_r2']:.3f}, "
            f"support=[{r['ind_own_q01']:.4f}, {r['ind_own_q99']:.4f}]"
        )
    lines.append("")
    lines.append("Interpretation note: lower `amihud_illiq` and `volatility` are better; higher `price_info` is better.")
    lines.append("This RF result is a flexible robustness check, not a causal estimate.")
    OUT_SUMMARY.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(DATA_PATH)
    df = complete_case_sample(raw)

    model_rows = []
    curve_rows = []
    for dep in DEP_VARS:
        row, curve = fit_one(df, dep)
        model_rows.append(row)
        curve_rows.append(curve)

    model_df = pd.DataFrame(model_rows)
    curve_df = pd.concat(curve_rows, ignore_index=True)
    score_df, score_row = build_score_and_plots(model_df, curve_df)

    model_df.to_csv(OUT_MODEL_SUMMARY, index=False)
    curve_df.to_csv(OUT_CURVE, index=False)
    score_df.to_csv(OUT_DIR / "rf_indexing_score_grid.csv", index=False)
    write_summary(model_df, score_row)

    print(f"Saved: {OUT_MODEL_SUMMARY}")
    print(f"Saved: {OUT_CURVE}")
    print(f"Saved: {OUT_SUMMARY}")
    print(f"Saved: {OUT_PLOT_SCORE}")
    print(f"Saved: {OUT_PLOT_CURVES}")


if __name__ == "__main__":
    main()
