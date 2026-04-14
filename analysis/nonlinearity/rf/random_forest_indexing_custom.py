#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import warnings

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DATA_PATH = Path("data/df_regression.csv")
OUT_DIR = Path("analysis/nonlinearity/rf")

DEP_VARS = ["amihud_illiq", "volatility", "price_info"]
MAIN = "ind_own"
CTRL_VARS = ["c_firm_size", "c_dollar_vol"]
DATE_COL = "date"

TRAIN_FRAC = 0.8
MAX_TRAIN_PER_DATE = 150
MAX_PD_PER_DATE = 60
MAX_ICE_PER_DATE = 20
N_TREES = 8
MAX_DEPTH = 4
MIN_SAMPLES_LEAF = 80
MAX_FEATURES = 3
BOOTSTRAP_FRAC = 0.5
N_THRESHOLD_CANDIDATES = 5
RANDOM_STATE = 42
GRID_SIZE = 220
BATCH_SIZE = 20

OUT_CURVE = OUT_DIR / "rf_indexing_curve_grid.csv"
OUT_MODEL_SUMMARY = OUT_DIR / "rf_indexing_model_summary.csv"
OUT_SCORE = OUT_DIR / "rf_indexing_score_grid.csv"
OUT_SUMMARY = OUT_DIR / "rf_indexing_summary.md"
OUT_PLOT_SCORE = OUT_DIR / "rf_indexing_score.svg"
OUT_PLOT_CURVES = OUT_DIR / "rf_relationship_all3.svg"
OUT_PLOT_IMPORTANCE = OUT_DIR / "rf_feature_importance.svg"
OUT_PLOT_PRED = OUT_DIR / "rf_observed_vs_predicted.svg"
OUT_PLOT_RESID = OUT_DIR / "rf_residual_diagnostics.svg"
OUT_PLOT_ICE = OUT_DIR / "rf_ice_curves_ind_own.svg"
OUT_PLOT_DIAGNOSTICS = OUT_DIR / "rf_publication_diagnostics.svg"


@dataclass
class Node:
    is_leaf: bool
    value: float
    feature: int | None = None
    threshold: float | None = None
    left: "Node | None" = None
    right: "Node | None" = None


class SimpleRegTree:
    def __init__(
        self,
        max_depth: int,
        min_samples_leaf: int,
        max_features: int,
        thresholds: list[np.ndarray],
        rng: np.random.Generator,
    ) -> None:
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.thresholds = thresholds
        self.rng = rng
        self.root: Node | None = None

    @staticmethod
    def _sse(y: np.ndarray) -> float:
        if y.size == 0:
            return 0.0
        s = float(y.sum())
        ss = float(np.dot(y, y))
        return ss - (s * s) / y.size

    def _build(self, X: np.ndarray, y: np.ndarray, idx: np.ndarray, depth: int) -> Node:
        y_node = y[idx]
        value = float(y_node.mean())
        if depth >= self.max_depth or idx.size <= 2 * self.min_samples_leaf:
            return Node(is_leaf=True, value=value)
        if np.var(y_node) <= 1e-12:
            return Node(is_leaf=True, value=value)

        p = X.shape[1]
        feat_candidates = self.rng.choice(p, size=min(self.max_features, p), replace=False)
        parent_sse = self._sse(y_node)
        best_gain = 0.0
        best_feat = None
        best_thr = None
        best_left = None
        best_right = None

        for feat in feat_candidates:
            xfeat = X[idx, feat]
            if np.allclose(xfeat.min(), xfeat.max()):
                continue
            for thr in self.thresholds[feat]:
                left_mask = xfeat <= thr
                n_left = int(left_mask.sum())
                n_right = idx.size - n_left
                if n_left < self.min_samples_leaf or n_right < self.min_samples_leaf:
                    continue
                y_left = y_node[left_mask]
                y_right = y_node[~left_mask]
                gain = parent_sse - (self._sse(y_left) + self._sse(y_right))
                if gain > best_gain:
                    best_gain = gain
                    best_feat = feat
                    best_thr = float(thr)
                    best_left = idx[left_mask]
                    best_right = idx[~left_mask]

        if best_feat is None or best_left is None or best_right is None:
            return Node(is_leaf=True, value=value)

        left_node = self._build(X, y, best_left, depth + 1)
        right_node = self._build(X, y, best_right, depth + 1)
        return Node(
            is_leaf=False,
            value=value,
            feature=best_feat,
            threshold=best_thr,
            left=left_node,
            right=right_node,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SimpleRegTree":
        idx = np.arange(len(y))
        self.root = self._build(X, y, idx, depth=0)
        return self

    def _predict_row(self, row: np.ndarray, node: Node) -> float:
        current = node
        while not current.is_leaf:
            if row[current.feature] <= current.threshold:
                current = current.left  # type: ignore[assignment]
            else:
                current = current.right  # type: ignore[assignment]
        return float(current.value)

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.root is None:
            raise RuntimeError("Tree has not been fitted.")
        out = np.empty(X.shape[0], dtype=float)
        for i in range(X.shape[0]):
            out[i] = self._predict_row(X[i], self.root)
        return out


class BaggedForest:
    def __init__(
        self,
        n_trees: int,
        max_depth: int,
        min_samples_leaf: int,
        max_features: int,
        bootstrap_frac: float,
        thresholds: list[np.ndarray],
        random_state: int,
    ) -> None:
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.bootstrap_frac = bootstrap_frac
        self.thresholds = thresholds
        self.random_state = random_state
        self.trees: list[SimpleRegTree] = []
        self.train_mean: float = np.nan

    def fit(self, X: np.ndarray, y: np.ndarray) -> "BaggedForest":
        rng = np.random.default_rng(self.random_state)
        n = len(y)
        boot_n = max(1, int(round(n * self.bootstrap_frac)))
        self.train_mean = float(y.mean())
        self.trees = []
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            for _ in range(self.n_trees):
                boot_idx = rng.integers(0, n, size=boot_n)
                tree = SimpleRegTree(
                    max_depth=self.max_depth,
                    min_samples_leaf=self.min_samples_leaf,
                    max_features=self.max_features,
                    thresholds=self.thresholds,
                    rng=np.random.default_rng(rng.integers(1, 2**32 - 1)),
                )
                tree.fit(X[boot_idx], y[boot_idx])
                self.trees.append(tree)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.trees:
            raise RuntimeError("Forest has not been fitted.")
        preds = np.column_stack([tree.predict(X) for tree in self.trees])
        return preds.mean(axis=1)


def complete_case_sample(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["cusip", DATE_COL, MAIN] + CTRL_VARS + DEP_VARS
    d = df[cols].copy()
    d[DATE_COL] = pd.to_datetime(d[DATE_COL])
    for c in [MAIN] + CTRL_VARS + DEP_VARS:
        d[c] = pd.to_numeric(d[c], errors="coerce")
        d.loc[~np.isfinite(d[c]), c] = np.nan
    d = d.dropna(subset=[MAIN] + CTRL_VARS + DEP_VARS).reset_index(drop=True)
    return d


def add_context_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    unique_dates = np.array(sorted(d[DATE_COL].unique()))
    date_map = {dt: i for i, dt in enumerate(unique_dates)}
    d["date_idx"] = d[DATE_COL].map(date_map).astype(float)

    for var in [MAIN] + CTRL_VARS:
        d[f"cusip_mean_{var}"] = d.groupby("cusip")[var].transform("mean")
        d[f"date_mean_{var}"] = d.groupby(DATE_COL)[var].transform("mean")
    return d


def split_dates(df: pd.DataFrame, train_frac: float, random_state: int) -> tuple[np.ndarray, np.ndarray]:
    unique_dates = np.array(sorted(df[DATE_COL].unique()))
    rng = np.random.default_rng(random_state)
    perm = rng.permutation(unique_dates)
    n_train = max(1, int(round(len(unique_dates) * train_frac)))
    train_dates = np.sort(perm[:n_train])
    eval_dates = np.sort(perm[n_train:])
    return train_dates, eval_dates


def sample_by_date(df: pd.DataFrame, max_per_date: int, random_state: int) -> pd.DataFrame:
    pieces = []
    for _, g in df.groupby(DATE_COL, sort=True):
        n = min(len(g), max_per_date)
        pieces.append(g.sample(n=n, random_state=random_state))
    sampled = pd.concat(pieces, ignore_index=True)
    return sampled.sort_values([DATE_COL, "cusip"]).reset_index(drop=True)


def build_feature_matrix(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    feature_cols = [
        MAIN,
        "c_firm_size",
        "c_dollar_vol",
        "date_idx",
        f"cusip_mean_{MAIN}",
        f"cusip_mean_c_firm_size",
        f"cusip_mean_c_dollar_vol",
        f"date_mean_{MAIN}",
        f"date_mean_c_firm_size",
        f"date_mean_c_dollar_vol",
    ]
    X = df[feature_cols].to_numpy(dtype=float)
    return X, feature_cols


def permutation_importance(
    forest: BaggedForest,
    X: np.ndarray,
    y: np.ndarray,
    base_r2: float,
    feature_names: list[str],
    random_state: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    rows = []
    for j, name in enumerate(feature_names):
        X_perm = X.copy()
        rng.shuffle(X_perm[:, j])
        perm_pred = forest.predict(X_perm)
        perm_r2 = r2_from_preds(y, perm_pred)
        rows.append(
            {
                "feature": name,
                "importance": float(base_r2 - perm_r2),
                "perm_r2": float(perm_r2),
                "base_r2": float(base_r2),
            }
        )
    return pd.DataFrame(rows).sort_values("importance", ascending=False).reset_index(drop=True)


def build_ice_curves(
    forest: BaggedForest,
    sample_df: pd.DataFrame,
    X_base: np.ndarray,
    grid: np.ndarray,
    feature_idx: int,
    max_lines: int,
) -> pd.DataFrame:
    if len(sample_df) == 0:
        return pd.DataFrame(columns=["dep_var", "line_id", "ind_own", "ice_level"])

    rng = np.random.default_rng(RANDOM_STATE + 101)
    n_lines = min(max_lines, len(sample_df))
    line_idx = np.sort(rng.choice(len(sample_df), size=n_lines, replace=False))
    rows = []
    for line_id, row_ix in enumerate(line_idx):
        base = X_base[row_ix : row_ix + 1].copy()
        own = float(sample_df.iloc[row_ix][MAIN])
        line_vals = []
        for x in grid:
            base[:, feature_idx] = x
            line_vals.append(float(forest.predict(base)[0]))
        line_vals = np.asarray(line_vals, dtype=float)
        ref_val = float(np.interp(own, grid, line_vals))
        rows.append(
            pd.DataFrame(
                {
                    "line_id": line_id,
                    "ind_own": grid,
                    "ice_level": line_vals - ref_val,
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def make_thresholds(X: np.ndarray) -> list[np.ndarray]:
    thresholds: list[np.ndarray] = []
    qs = np.linspace(0.12, 0.88, N_THRESHOLD_CANDIDATES)
    for j in range(X.shape[1]):
        col = X[:, j]
        vals = np.unique(np.quantile(col, qs))
        if vals.size == 0:
            vals = np.array([float(np.median(col))], dtype=float)
        thresholds.append(vals.astype(float))
    return thresholds


def fit_one(df: pd.DataFrame, dep_var: str) -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_dates, eval_dates = split_dates(df, TRAIN_FRAC, RANDOM_STATE)
    train_full = df[df[DATE_COL].isin(train_dates)].copy().reset_index(drop=True)
    eval_full = df[df[DATE_COL].isin(eval_dates)].copy().reset_index(drop=True)

    train = sample_by_date(train_full, MAX_TRAIN_PER_DATE, RANDOM_STATE)
    eval_df = eval_full.copy()
    pd_sample = sample_by_date(train_full, MAX_PD_PER_DATE, RANDOM_STATE + 11)
    ice_sample = sample_by_date(eval_full, MAX_ICE_PER_DATE, RANDOM_STATE + 23)

    X_train, feature_cols = build_feature_matrix(train)
    y_train = train[dep_var].to_numpy(dtype=float)
    thresholds = make_thresholds(X_train)

    forest = BaggedForest(
        n_trees=N_TREES,
        max_depth=MAX_DEPTH,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        max_features=MAX_FEATURES,
        bootstrap_frac=BOOTSTRAP_FRAC,
        thresholds=thresholds,
        random_state=RANDOM_STATE,
    ).fit(X_train, y_train)

    pred_train = forest.predict(X_train)
    X_eval, _ = build_feature_matrix(eval_df)
    y_eval = eval_df[dep_var].to_numpy(dtype=float)
    pred_eval = forest.predict(X_eval)
    eval_r2 = float(r2_from_preds(y_eval, pred_eval))
    train_r2 = float(r2_from_preds(y_train, pred_train))
    imp_df = permutation_importance(
        forest,
        X_eval,
        y_eval,
        eval_r2,
        feature_cols,
        random_state=RANDOM_STATE + 19,
    )

    X_pd, _ = build_feature_matrix(pd_sample)
    X_ice, _ = build_feature_matrix(ice_sample)
    xgrid = np.linspace(
        float(train[MAIN].quantile(0.01)),
        float(train[MAIN].quantile(0.99)),
        GRID_SIZE,
    )
    x_ref = float(train[MAIN].quantile(0.50))
    pd_curve = partial_dependence_curve(forest, X_pd, xgrid, feature_idx=0)
    pd_ref = float(partial_dependence_curve(forest, X_pd, np.array([x_ref]), feature_idx=0)[0])
    delta = pd_curve - pd_ref
    ice_df = build_ice_curves(forest, ice_sample, X_ice, xgrid, feature_idx=0, max_lines=24)

    row = {
        "dep_var": dep_var,
        "train_nobs": int(len(train)),
        "eval_nobs": int(len(eval_df)),
        "pd_nobs": int(len(pd_sample)),
        "entities_train": int(train["cusip"].nunique()),
        "dates_train": int(train[DATE_COL].nunique()),
        "entities_eval": int(eval_df["cusip"].nunique()),
        "dates_eval": int(eval_df[DATE_COL].nunique()),
        "train_r2": train_r2,
        "eval_r2": eval_r2,
        "ind_own_min": float(train[MAIN].min()),
        "ind_own_q01": float(train[MAIN].quantile(0.01)),
        "ind_own_q05": float(train[MAIN].quantile(0.05)),
        "ind_own_q50": float(train[MAIN].quantile(0.50)),
        "ind_own_q95": float(train[MAIN].quantile(0.95)),
        "ind_own_q99": float(train[MAIN].quantile(0.99)),
        "ind_own_max": float(train[MAIN].max()),
    }

    curve = pd.DataFrame(
        {
            "dep_var": dep_var,
            "ind_own": xgrid,
            "pred_level": pd_curve,
            "delta_from_median": delta,
        }
    )
    pred_df = pd.DataFrame(
        {
            "dep_var": dep_var,
            "sample": "train",
            "y_true": y_train,
            "y_pred": pred_train,
            "residual": y_train - pred_train,
        }
    )
    pred_eval_df = pd.DataFrame(
        {
            "dep_var": dep_var,
            "sample": "eval",
            "y_true": y_eval,
            "y_pred": pred_eval,
            "residual": y_eval - pred_eval,
        }
    )
    pred_df = pd.concat([pred_df, pred_eval_df], ignore_index=True)
    return row, curve, imp_df, pred_df, ice_df


def r2_from_preds(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot <= 1e-12:
        return np.nan
    return 1.0 - ss_res / ss_tot


def partial_dependence_curve(
    forest: BaggedForest,
    X_base: np.ndarray,
    grid: np.ndarray,
    feature_idx: int,
) -> np.ndarray:
    out = np.empty(len(grid), dtype=float)
    for start in range(0, len(grid), BATCH_SIZE):
        stop = min(start + BATCH_SIZE, len(grid))
        batch = grid[start:stop]
        for i, val in enumerate(batch, start=start):
            X_tmp = X_base.copy()
            X_tmp[:, feature_idx] = val
            out[i] = float(forest.predict(X_tmp).mean())
    return out


def zscore(v: np.ndarray) -> np.ndarray:
    s = float(np.std(v))
    if not np.isfinite(s) or s <= 1e-12:
        return np.zeros_like(v, dtype=float)
    return (v - float(np.mean(v))) / s


def build_score_and_plots(curve_df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    xgrid = np.sort(curve_df["ind_own"].unique())
    curves = {}
    for dep in DEP_VARS:
        curves[dep] = (
            curve_df.loc[curve_df["dep_var"] == dep]
            .sort_values("ind_own")["delta_from_median"]
            .to_numpy(dtype=float)
        )

    score = -zscore(curves["amihud_illiq"]) - zscore(curves["volatility"]) + zscore(curves["price_info"])
    best_idx = int(np.argmax(score))
    best_x = float(xgrid[best_idx])
    score_q95 = float(np.quantile(score, 0.95))
    top_mask = score >= score_q95
    top_x = xgrid[top_mask]
    top_lo = float(top_x.min()) if len(top_x) else np.nan
    top_hi = float(top_x.max()) if len(top_x) else np.nan

    score_df = pd.DataFrame(
        {
            "ind_own": xgrid,
            "score_total": score,
            "delta_amihud": curves["amihud_illiq"],
            "delta_volatility": curves["volatility"],
            "delta_price_info": curves["price_info"],
            "is_top5pct_score": top_mask.astype(int),
        }
    )
    score_df = pd.concat(
        [
            score_df,
            pd.DataFrame(
                [
                    {
                        "ind_own": best_x,
                        "score_total": float(score[best_idx]),
                        "delta_amihud": np.nan,
                        "delta_volatility": np.nan,
                        "delta_price_info": np.nan,
                        "is_top5pct_score": 2,
                    },
                    {
                        "ind_own": top_lo,
                        "score_total": score_q95,
                        "delta_amihud": np.nan,
                        "delta_volatility": np.nan,
                        "delta_price_info": np.nan,
                        "is_top5pct_score": 3,
                    },
                    {
                        "ind_own": top_hi,
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
    if np.isfinite(top_lo) and np.isfinite(top_hi):
        ax.axvspan(top_lo, top_hi, color="#9ecae1", alpha=0.32, lw=0, label="Top 5% score range")
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

    return score_df, {"best_ind_own": best_x, "top_lo": top_lo, "top_hi": top_hi, "score_q95": score_q95}


def plot_feature_importance(importance_df: pd.DataFrame) -> None:
    dep_order = DEP_VARS
    fig, axes = plt.subplots(1, len(dep_order), figsize=(16.5, 4.8), sharex=False)
    for ax, dep in zip(axes, dep_order):
        d = importance_df[importance_df["dep_var"] == dep].sort_values("importance", ascending=True)
        ax.barh(d["feature"], d["importance"], color="#4c78a8")
        ax.set_title(dep.replace("_", " ").title(), fontsize=11)
        ax.grid(axis="x", alpha=0.22, linestyle=":")
        ax.set_xlabel("Permutation importance")
        if ax is axes[0]:
            ax.set_ylabel("Feature")
    fig.suptitle("Random Forest Variable Importance", fontsize=14, y=0.98)
    fig.tight_layout()
    fig.savefig(OUT_PLOT_IMPORTANCE, format="svg", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_residual_diagnostics(pred_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, len(DEP_VARS), figsize=(17.2, 8.2))
    titles = {
        "amihud_illiq": "Illiquidity",
        "volatility": "Volatility",
        "price_info": "Price Informativeness",
    }
    for j, dep in enumerate(DEP_VARS):
        d = pred_df[pred_df["dep_var"] == dep]
        train = d[d["sample"] == "train"]
        eval_ = d[d["sample"] == "eval"]

        ax = axes[0, j]
        ax.hist(eval_["residual"], bins=40, color="#6baed6", alpha=0.85, density=True)
        ax.axvline(0.0, color="0.25", linestyle="--", linewidth=1.0)
        ax.set_title(titles[dep], fontsize=11)
        ax.set_xlabel("Residual")
        ax.set_ylabel("Density")
        ax.grid(axis="y", alpha=0.2, linestyle=":")

        ax = axes[1, j]
        ax.scatter(train["y_pred"], train["residual"], s=7, alpha=0.12, color="#9ecae1", label="Train")
        ax.scatter(eval_["y_pred"], eval_["residual"], s=9, alpha=0.18, color="#3182bd", label="Eval")
        ax.axhline(0.0, color="0.25", linestyle="--", linewidth=1.0)
        ax.set_title(titles[dep], fontsize=11)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Residual")
        ax.grid(alpha=0.2, linestyle=":")
        if j == 0:
            ax.legend(frameon=False, loc="best")
    fig.suptitle("Random Forest Residual Diagnostics", fontsize=14, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(OUT_PLOT_RESID, format="svg", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_ice_curves(ice_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, len(DEP_VARS), figsize=(16.5, 4.8), sharey=False)
    meta = {
        "amihud_illiq": ("Illiquidity", "#8c2d04"),
        "volatility": ("Volatility", "#08519c"),
        "price_info": ("Price Informativeness", "#006d2c"),
    }
    for ax, dep in zip(axes, DEP_VARS):
        d = ice_df[ice_df["dep_var"] == dep]
        for _, g in d.groupby("line_id"):
            ax.plot(g["ind_own"], g["ice_level"], color=meta[dep][1], alpha=0.17, linewidth=1.0)
        pdp = d.groupby("ind_own", as_index=False)["ice_level"].mean()
        ax.plot(pdp["ind_own"], pdp["ice_level"], color="#111111", linewidth=2.2, label="Average ICE")
        ax.axhline(0.0, color="0.35", linewidth=0.9, linestyle=":")
        ax.set_title(meta[dep][0], fontsize=11)
        ax.set_xlabel("ind_own")
        ax.grid(axis="y", alpha=0.2, linestyle=":")
    axes[0].set_ylabel("ICE effect relative to own median")
    fig.suptitle("Random Forest ICE Curves for Index Ownership", fontsize=14, y=0.98)
    fig.tight_layout()
    fig.savefig(OUT_PLOT_ICE, format="svg", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_publication_diagnostics(
    importance_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    score_df: pd.DataFrame,
    ice_df: pd.DataFrame,
) -> None:
    dep = "price_info"
    fig, axes = plt.subplots(2, 2, figsize=(15.4, 11.0))

    d_imp = importance_df[importance_df["dep_var"] == dep].sort_values("importance", ascending=True)
    axes[0, 0].barh(d_imp["feature"], d_imp["importance"], color="#4c78a8")
    axes[0, 0].set_title("Variable Importance", fontsize=12)
    axes[0, 0].set_xlabel("Permutation importance")
    axes[0, 0].grid(axis="x", alpha=0.2, linestyle=":")

    d_pred = pred_df[pred_df["dep_var"] == dep]
    tr = d_pred[d_pred["sample"] == "train"]
    ev = d_pred[d_pred["sample"] == "eval"]
    lo = float(min(d_pred["y_true"].min(), d_pred["y_pred"].min()))
    hi = float(max(d_pred["y_true"].max(), d_pred["y_pred"].max()))
    axes[0, 1].scatter(tr["y_true"], tr["y_pred"], s=7, alpha=0.16, color="#9ecae1", label="Train")
    axes[0, 1].scatter(ev["y_true"], ev["y_pred"], s=9, alpha=0.22, color="#3182bd", label="Eval")
    axes[0, 1].plot([lo, hi], [lo, hi], color="0.25", linestyle="--", linewidth=1.0)
    axes[0, 1].set_title("Observed vs Predicted", fontsize=12)
    axes[0, 1].set_xlabel("Observed")
    axes[0, 1].set_ylabel("Predicted")
    axes[0, 1].legend(frameon=False, loc="best")
    axes[0, 1].grid(alpha=0.2, linestyle=":")

    axes[1, 0].hist(ev["residual"], bins=40, color="#6baed6", alpha=0.85, density=True)
    axes[1, 0].axvline(0.0, color="0.25", linestyle="--", linewidth=1.0)
    axes[1, 0].set_title("Residual Distribution", fontsize=12)
    axes[1, 0].set_xlabel("Residual")
    axes[1, 0].set_ylabel("Density")
    axes[1, 0].grid(axis="y", alpha=0.2, linestyle=":")

    d_ice = ice_df[ice_df["dep_var"] == dep]
    for _, g in d_ice.groupby("line_id"):
        axes[1, 1].plot(g["ind_own"], g["ice_level"], color="#006d2c", alpha=0.15, linewidth=1.0)
    pdp = d_ice.groupby("ind_own", as_index=False)["ice_level"].mean()
    axes[1, 1].plot(pdp["ind_own"], pdp["ice_level"], color="#111111", linewidth=2.2)
    score_base = score_df[score_df["is_top5pct_score"].isin([0, 1])].sort_values("ind_own")
    axes[1, 1].twinx().plot(score_base["ind_own"], score_base["score_total"], color="#d62728", alpha=0.35, linewidth=1.3)
    axes[1, 1].set_title("ICE and Score Shape", fontsize=12)
    axes[1, 1].set_xlabel("ind_own")
    axes[1, 1].set_ylabel("ICE effect")
    axes[1, 1].grid(axis="y", alpha=0.2, linestyle=":")

    fig.suptitle("Random Forest Publication Diagnostics", fontsize=15, y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(OUT_PLOT_DIAGNOSTICS, format="svg", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_predicted_vs_observed(pred_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, len(DEP_VARS), figsize=(16.5, 4.8))
    colors = {"train": "#9ecae1", "eval": "#3182bd"}
    for ax, dep in zip(axes, DEP_VARS):
        d = pred_df[pred_df["dep_var"] == dep]
        train = d[d["sample"] == "train"]
        eval_ = d[d["sample"] == "eval"]
        lo = float(min(d["y_true"].min(), d["y_pred"].min()))
        hi = float(max(d["y_true"].max(), d["y_pred"].max()))
        ax.scatter(train["y_true"], train["y_pred"], s=8, alpha=0.18, c=colors["train"], label="Train")
        ax.scatter(eval_["y_true"], eval_["y_pred"], s=10, alpha=0.25, c=colors["eval"], label="Eval")
        ax.plot([lo, hi], [lo, hi], color="0.25", linestyle="--", linewidth=1.0)
        ax.set_title(dep.replace("_", " ").title(), fontsize=11)
        ax.set_xlabel("Observed")
        ax.set_ylabel("Predicted")
        ax.grid(alpha=0.20, linestyle=":")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="upper center", ncol=2)
    fig.suptitle("Random Forest Observed vs Predicted", fontsize=14, y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(OUT_PLOT_PRED, format="svg", dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_summary(model_df: pd.DataFrame, score_meta: dict) -> None:
    lines: list[str] = []
    lines.append("# Random Forest Nonlinearity: How Much Indexing Is Good?")
    lines.append("")
    lines.append("Specification:")
    lines.append("- Bagged regression trees estimated in pure Python/Numpy because `scikit-learn` is unavailable in this environment.")
    lines.append("- Main predictor: `ind_own`.")
    lines.append("- Controls: `c_firm_size`, `c_dollar_vol`, date index, and firm/date mean context variables.")
    lines.append("- The model is a flexible nonparametric robustness check, not a causal design.")
    lines.append("")
    lines.append("## Sample and Fit")
    for _, r in model_df.iterrows():
        lines.append(
            f"- `{r['dep_var']}`: train n={int(r['train_nobs'])}, eval n={int(r['eval_nobs'])}, "
            f"pd n={int(r['pd_nobs'])}, train R2={r['train_r2']:.3f}, eval R2={r['eval_r2']:.3f}"
        )
    lines.append("")
    lines.append("## Composite Answer")
    lines.append(f"- Composite optimum indexing level: `ind_own ≈ {score_meta['best_ind_own']:.4f}`")
    lines.append(f"- High-quality range (top 5% score): `[{score_meta['top_lo']:.4f}, {score_meta['top_hi']:.4f}]`")
    lines.append("")
    lines.append("## Reading the Curves")
    lines.append("- Lower `amihud_illiq` and `volatility` are better.")
    lines.append("- Higher `price_info` is better.")
    lines.append("- The score combines the three standardized partial-dependence curves with the appropriate signs.")
    lines.append("")
    lines.append("## Diagnostic Plots")
    lines.append("- `rf_feature_importance.svg`: permutation importance by outcome.")
    lines.append("- `rf_observed_vs_predicted.svg`: observed vs fitted values for train and eval samples.")
    lines.append("- `rf_residual_diagnostics.svg`: residual histograms and residual-vs-fitted plots.")
    lines.append("- `rf_ice_curves_ind_own.svg`: individual conditional expectation curves for `ind_own`.")
    lines.append("- `rf_publication_diagnostics.svg`: compact appendix-style panel for the RF model.")
    OUT_SUMMARY.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(DATA_PATH)
    df = complete_case_sample(raw)
    df = add_context_features(df)

    model_rows = []
    curve_rows = []
    importance_rows = []
    pred_rows = []
    ice_rows = []
    for dep in DEP_VARS:
        row, curve, imp_df, pred_df, ice_df = fit_one(df, dep)
        model_rows.append(row)
        curve_rows.append(curve)
        importance_rows.append(imp_df.assign(dep_var=dep))
        pred_rows.append(pred_df)
        ice_rows.append(ice_df.assign(dep_var=dep))

    model_df = pd.DataFrame(model_rows)
    curve_df = pd.concat(curve_rows, ignore_index=True)
    importance_df = pd.concat(importance_rows, ignore_index=True)
    pred_df = pd.concat(pred_rows, ignore_index=True)
    ice_df = pd.concat(ice_rows, ignore_index=True)
    score_df, score_meta = build_score_and_plots(curve_df)

    model_df.to_csv(OUT_MODEL_SUMMARY, index=False)
    curve_df.to_csv(OUT_CURVE, index=False)
    score_df.to_csv(OUT_SCORE, index=False)
    importance_df.to_csv(OUT_DIR / "rf_feature_importance.csv", index=False)
    pred_df.to_csv(OUT_DIR / "rf_observed_vs_predicted.csv", index=False)
    ice_df.to_csv(OUT_DIR / "rf_ice_curves_ind_own.csv", index=False)
    plot_feature_importance(importance_df)
    plot_predicted_vs_observed(pred_df)
    plot_residual_diagnostics(pred_df)
    plot_ice_curves(ice_df)
    plot_publication_diagnostics(importance_df, pred_df, score_df, ice_df)
    write_summary(model_df, score_meta)

    print(f"Saved: {OUT_MODEL_SUMMARY}")
    print(f"Saved: {OUT_CURVE}")
    print(f"Saved: {OUT_SCORE}")
    print(f"Saved: {OUT_DIR / 'rf_feature_importance.csv'}")
    print(f"Saved: {OUT_DIR / 'rf_observed_vs_predicted.csv'}")
    print(f"Saved: {OUT_DIR / 'rf_ice_curves_ind_own.csv'}")
    print(f"Saved: {OUT_SUMMARY}")
    print(f"Saved: {OUT_PLOT_SCORE}")
    print(f"Saved: {OUT_PLOT_CURVES}")
    print(f"Saved: {OUT_PLOT_IMPORTANCE}")
    print(f"Saved: {OUT_PLOT_PRED}")
    print(f"Saved: {OUT_PLOT_RESID}")
    print(f"Saved: {OUT_PLOT_ICE}")
    print(f"Saved: {OUT_PLOT_DIAGNOSTICS}")


if __name__ == "__main__":
    main()
