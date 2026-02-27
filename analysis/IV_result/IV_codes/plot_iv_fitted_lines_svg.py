#!/usr/bin/env python3
import math
from pathlib import Path

import numpy as np

import panel_iv_rank_cutoff as piv

OUT_PATH = Path("analysis/iv_fitted_lines_by_depvar.svg")
SEED = 42
MAX_POINTS = 1800


def sample_idx(n, k, rng):
    if n <= k:
        return np.arange(n)
    return np.sort(rng.choice(n, size=k, replace=False))


def clip_quantile(x, q=0.01):
    lo = float(np.quantile(x, q))
    hi = float(np.quantile(x, 1 - q))
    return lo, hi


def xmap(x, xmin, xmax, left, width):
    return left + (x - xmin) / (xmax - xmin) * width


def ymap(y, ymin, ymax, top, height):
    return top + (ymax - y) / (ymax - ymin) * height


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def main():
    rank_map, _ = piv.load_rank_cutoff_map(piv.RANK_PATH)
    rng = np.random.default_rng(SEED)

    panels = []

    for dep in piv.DEP_VARS:
        y, x, z, w, firm_ids, time_ids = piv.prepare_dataset(dep, rank_map, piv.CUTOFF, piv.BANDWIDTH)
        M = np.column_stack([y, x, z, w])
        M_dm = piv.two_way_demean(M, firm_ids, time_ids)

        y_dm = M_dm[:, 0]
        x_dm = M_dm[:, 1]
        z_dm = M_dm[:, 2]
        w_dm = M_dm[:, 3:]

        beta, _ = piv.iv_2sls_clustered(y_dm, x_dm, z_dm, w_dm, firm_ids)
        b = float(beta[0])
        g = beta[1:]

        idx = sample_idx(len(y_dm), MAX_POINTS, rng)
        xs = x_dm[idx]
        ys = y_dm[idx]

        xlo, xhi = clip_quantile(x_dm, 0.01)
        ylo = float(np.quantile(y_dm, 0.01))
        yhi = float(np.quantile(y_dm, 0.99))

        y_line_lo = b * xlo
        y_line_hi = b * xhi

        if not np.isfinite(ylo) or not np.isfinite(yhi) or yhi <= ylo:
            ylo, yhi = float(np.min(y_dm)), float(np.max(y_dm))
        if not np.isfinite(xlo) or not np.isfinite(xhi) or xhi <= xlo:
            xlo, xhi = float(np.min(x_dm)), float(np.max(x_dm))

        ylo = min(ylo, y_line_lo, y_line_hi)
        yhi = max(yhi, y_line_lo, y_line_hi)

        panels.append(
            {
                "dep": dep,
                "beta": b,
                "n": int(len(y_dm)),
                "x": xs,
                "y": ys,
                "xlo": xlo,
                "xhi": xhi,
                "ylo": ylo,
                "yhi": yhi,
            }
        )

    W = 1300
    H = 450
    MT = 58
    MB = 58
    ML = 60
    MR = 30
    gap = 34
    n = len(panels)
    pw = (W - ML - MR - gap * (n - 1)) / n
    ph = H - MT - MB

    parts = []
    parts.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">')
    parts.append("<style>")
    parts.append("text{font-family:Arial,Helvetica,sans-serif;fill:#111827}")
    parts.append(".title{font-size:20px;font-weight:700}")
    parts.append(".axis{stroke:#374151;stroke-width:1.2}")
    parts.append(".grid{stroke:#e5e7eb;stroke-width:1}")
    parts.append(".pt{fill:#9ca3af;fill-opacity:.35}")
    parts.append(".fit{stroke:#dc2626;stroke-width:2.4}")
    parts.append(".lbl{font-size:13px}")
    parts.append(".tick{font-size:11px;fill:#4b5563}")
    parts.append("</style>")
    parts.append(f'<rect x="0" y="0" width="{W}" height="{H}" fill="white"/>')
    parts.append(f'<text x="{W/2}" y="32" text-anchor="middle" class="title">Global Cutoff-IV with two-way FE and Robust (non-clustered) SEs</text>')

    for i, p in enumerate(panels):
        left = ML + i * (pw + gap)
        top = MT

        # frame
        parts.append(f'<rect x="{left:.2f}" y="{top:.2f}" width="{pw:.2f}" height="{ph:.2f}" fill="none" class="axis"/>')

        # grid x=0,y=0 if inside
        if p["xlo"] < 0 < p["xhi"]:
            x0 = xmap(0.0, p["xlo"], p["xhi"], left, pw)
            parts.append(f'<line x1="{x0:.2f}" y1="{top:.2f}" x2="{x0:.2f}" y2="{top+ph:.2f}" class="grid"/>')
        if p["ylo"] < 0 < p["yhi"]:
            y0 = ymap(0.0, p["ylo"], p["yhi"], top, ph)
            parts.append(f'<line x1="{left:.2f}" y1="{y0:.2f}" x2="{left+pw:.2f}" y2="{y0:.2f}" class="grid"/>')

        # scatter
        for xv, yv in zip(p["x"], p["y"]):
            if xv < p["xlo"] or xv > p["xhi"] or yv < p["ylo"] or yv > p["yhi"]:
                continue
            xp = xmap(float(xv), p["xlo"], p["xhi"], left, pw)
            yp = ymap(float(yv), p["ylo"], p["yhi"], top, ph)
            parts.append(f'<circle cx="{xp:.2f}" cy="{yp:.2f}" r="1.8" class="pt"/>')

        # fitted line: y = beta * x (demeaned scale)
        x1, x2 = p["xlo"], p["xhi"]
        y1, y2 = p["beta"] * x1, p["beta"] * x2
        xp1 = xmap(x1, p["xlo"], p["xhi"], left, pw)
        xp2 = xmap(x2, p["xlo"], p["xhi"], left, pw)
        yp1 = ymap(y1, p["ylo"], p["yhi"], top, ph)
        yp2 = ymap(y2, p["ylo"], p["yhi"], top, ph)
        parts.append(f'<line x1="{xp1:.2f}" y1="{yp1:.2f}" x2="{xp2:.2f}" y2="{yp2:.2f}" class="fit"/>')

        # labels
        parts.append(f'<text x="{left + pw/2:.2f}" y="{top - 10:.2f}" text-anchor="middle" class="lbl">{esc(p["dep"])}</text>')
        parts.append(f'<text x="{left + 8:.2f}" y="{top + 16:.2f}" class="tick">beta={p["beta"]:.4f}</text>')
        parts.append(f'<text x="{left + 8:.2f}" y="{top + 30:.2f}" class="tick">n={p["n"]}</text>')

        # x ticks
        for t in [p["xlo"], 0.5 * (p["xlo"] + p["xhi"]), p["xhi"]]:
            tx = xmap(t, p["xlo"], p["xhi"], left, pw)
            parts.append(f'<line x1="{tx:.2f}" y1="{top+ph:.2f}" x2="{tx:.2f}" y2="{top+ph+5:.2f}" class="axis"/>')
            parts.append(f'<text x="{tx:.2f}" y="{top+ph+18:.2f}" text-anchor="middle" class="tick">{t:.2f}</text>')

    parts.append(f'<text x="{W/2:.2f}" y="{H-16:.2f}" text-anchor="middle" class="lbl">Demeaned ind_own (x-axis) vs demeaned dependent variable (y-axis)</text>')
    parts.append("</svg>")

    OUT_PATH.write_text("\n".join(parts), encoding="utf-8")
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
