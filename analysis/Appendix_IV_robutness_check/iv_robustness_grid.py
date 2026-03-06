#!/usr/bin/env python3
import csv
import math
from pathlib import Path

import numpy as np

import panel_iv_rank_cutoff as piv


OUT_CSV = Path("analysis/iv_robustness_grid_results.csv")
OUT_F_PLOT = Path("analysis/iv_robustness_first_stage_f.svg")
OUT_BETA_PLOT = Path("analysis/iv_robustness_beta.svg")

CUTOFFS = [1000.0, 2000.0]
BANDWIDTHS = [50.0, 100.0, 150.0, 200.0, 300.0, 500.0, None]


def bw_label(bw):
    return "global" if bw is None else str(int(bw))


def esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def render_metric_plot(rows, metric_key, title, y_label, out_path: Path):
    dep_vars = piv.DEP_VARS
    cutoffs = CUTOFFS
    categories = [bw_label(b) for b in BANDWIDTHS]
    cat_pos = {c: i for i, c in enumerate(categories)}

    W, H = 1360, 460
    ML, MR, MT, MB = 62, 32, 60, 64
    GAP = 30
    n_pan = len(dep_vars)
    pw = (W - ML - MR - GAP * (n_pan - 1)) / n_pan
    ph = H - MT - MB

    def x_to_px(cat, left):
        i = cat_pos[cat]
        if len(categories) == 1:
            return left + pw / 2
        return left + (i / (len(categories) - 1)) * pw

    vals = [r[metric_key] for r in rows if math.isfinite(r[metric_key])]
    if not vals:
        ymin, ymax = -1.0, 1.0
    else:
        ymin = min(vals)
        ymax = max(vals)
        if ymax <= ymin:
            ymax = ymin + 1.0
        pad = 0.10 * (ymax - ymin)
        ymin -= pad
        ymax += pad

    def y_to_px(y):
        return MT + (ymax - y) / (ymax - ymin) * ph

    parts = []
    parts.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">')
    parts.append("<style>")
    parts.append("text{font-family:Arial,Helvetica,sans-serif;fill:#111827}")
    parts.append(".title{font-size:20px;font-weight:700}")
    parts.append(".axis{stroke:#374151;stroke-width:1.2}")
    parts.append(".grid{stroke:#e5e7eb;stroke-width:1}")
    parts.append(".lineA{stroke:#2563eb;stroke-width:2.2;fill:none}")
    parts.append(".lineB{stroke:#dc2626;stroke-width:2.2;fill:none}")
    parts.append(".ptA{fill:#1d4ed8}")
    parts.append(".ptB{fill:#b91c1c}")
    parts.append(".lbl{font-size:13px}")
    parts.append(".tick{font-size:11px;fill:#4b5563}")
    parts.append("</style>")
    parts.append(f'<rect x="0" y="0" width="{W}" height="{H}" fill="white"/>')
    parts.append(f'<text x="{W/2}" y="34" text-anchor="middle" class="title">{esc(title)}</text>')

    # global legend
    parts.append(f'<line x1="{W-260}" y1="24" x2="{W-235}" y2="24" class="lineA"/>')
    parts.append(f'<text x="{W-228}" y="28" class="tick">cutoff=1000</text>')
    parts.append(f'<line x1="{W-145}" y1="24" x2="{W-120}" y2="24" class="lineB"/>')
    parts.append(f'<text x="{W-113}" y="28" class="tick">cutoff=2000</text>')

    if ymin < 0 < ymax:
        y0 = y_to_px(0.0)
        parts.append(f'<line x1="{ML}" y1="{y0:.2f}" x2="{W-MR}" y2="{y0:.2f}" stroke="#9ca3af" stroke-width="1" stroke-dasharray="4 4"/>')

    # Build quick lookup
    lookup = {(r["dep_var"], r["cutoff"], r["bw_label"]): r[metric_key] for r in rows}

    for p, dep in enumerate(dep_vars):
        left = ML + p * (pw + GAP)
        parts.append(f'<rect x="{left:.2f}" y="{MT:.2f}" width="{pw:.2f}" height="{ph:.2f}" fill="none" class="axis"/>')
        parts.append(f'<text x="{left + pw/2:.2f}" y="{MT-10:.2f}" text-anchor="middle" class="lbl">{esc(dep)}</text>')

        for frac in [0.0, 0.5, 1.0]:
            yv = ymin + frac * (ymax - ymin)
            yp = y_to_px(yv)
            parts.append(f'<line x1="{left:.2f}" y1="{yp:.2f}" x2="{left+pw:.2f}" y2="{yp:.2f}" class="grid"/>')
            parts.append(f'<text x="{left-6:.2f}" y="{yp+4:.2f}" text-anchor="end" class="tick">{yv:.2f}</text>')

        for cutoff, line_cls, pt_cls in [(1000.0, "lineA", "ptA"), (2000.0, "lineB", "ptB")]:
            pts = []
            for c in categories:
                v = lookup.get((dep, cutoff, c), float("nan"))
                if not math.isfinite(v):
                    continue
                pts.append((x_to_px(c, left), y_to_px(v), c, v))
            if len(pts) >= 2:
                d = "M " + " L ".join(f"{x:.2f} {y:.2f}" for x, y, _, _ in pts)
                parts.append(f'<path d="{d}" class="{line_cls}"/>')
            for x, y, _, _ in pts:
                parts.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="3.2" class="{pt_cls}"/>')

        for c in categories:
            xp = x_to_px(c, left)
            parts.append(f'<line x1="{xp:.2f}" y1="{MT+ph:.2f}" x2="{xp:.2f}" y2="{MT+ph+4:.2f}" class="axis"/>')
            parts.append(f'<text x="{xp:.2f}" y="{MT+ph+18:.2f}" text-anchor="middle" class="tick">{esc(c)}</text>')

    parts.append(f'<text x="{24}" y="{MT + ph/2:.2f}" transform="rotate(-90 24 {MT + ph/2:.2f})" text-anchor="middle" class="lbl">{esc(y_label)}</text>')
    parts.append(f'<text x="{W/2:.2f}" y="{H-16:.2f}" text-anchor="middle" class="lbl">Bandwidth</text>')
    parts.append("</svg>")

    out_path.write_text("\n".join(parts), encoding="utf-8")


def main():
    rank_map, dup_count = piv.load_rank_cutoff_map(piv.RANK_PATH)
    print(f"Loaded rank map: {len(rank_map)} keys, duplicates resolved: {dup_count}")

    out_rows = []
    for cutoff in CUTOFFS:
        for bw in BANDWIDTHS:
            print(f"Running grid: cutoff={cutoff}, bandwidth={bw_label(bw)}")
            for dep in piv.DEP_VARS:
                res = piv.run_one(dep, rank_map, cutoff, bw)
                out_rows.append(
                    {
                        "dep_var": dep,
                        "cutoff": cutoff,
                        "bandwidth": bw,
                        "bw_label": bw_label(bw),
                        "nobs": res.nobs,
                        "n_firms": res.n_firms,
                        "n_dates": res.n_dates,
                        "n_clusters": res.n_clusters,
                        "beta_endog": res.beta_endog,
                        "se_endog": res.se_endog,
                        "t_endog": res.t_endog,
                        "p_endog": res.p_endog,
                        "first_stage_f": res.first_stage_f,
                        "first_stage_p": res.first_stage_p,
                    }
                )

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "dep_var",
                "cutoff",
                "bandwidth",
                "bw_label",
                "nobs",
                "n_firms",
                "n_dates",
                "n_clusters",
                "beta_endog",
                "se_endog",
                "t_endog",
                "p_endog",
                "first_stage_f",
                "first_stage_p",
            ]
        )
        for r in out_rows:
            writer.writerow(
                [
                    r["dep_var"],
                    r["cutoff"],
                    r["bandwidth"],
                    r["bw_label"],
                    r["nobs"],
                    r["n_firms"],
                    r["n_dates"],
                    r["n_clusters"],
                    r["beta_endog"],
                    r["se_endog"],
                    r["t_endog"],
                    r["p_endog"],
                    r["first_stage_f"],
                    r["first_stage_p"],
                ]
            )

    render_metric_plot(
        out_rows,
        metric_key="first_stage_f",
        title="First-Stage F Across Bandwidths",
        y_label="First-stage F",
        out_path=OUT_F_PLOT,
    )
    render_metric_plot(
        out_rows,
        metric_key="beta_endog",
        title="IV Coefficient on ind_own Across Bandwidths",
        y_label="2SLS coefficient",
        out_path=OUT_BETA_PLOT,
    )

    print(f"Saved grid results: {OUT_CSV}")
    print(f"Saved plot: {OUT_F_PLOT}")
    print(f"Saved plot: {OUT_BETA_PLOT}")


if __name__ == "__main__":
    main()
