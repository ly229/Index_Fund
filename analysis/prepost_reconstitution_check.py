#!/usr/bin/env python3
import csv
from datetime import datetime
from pathlib import Path

import numpy as np

import panel_iv_rank_cutoff as piv

MAIN_PATH = Path("data/df_regression.csv")
RANK_PATH = Path("data/Russell_3_rank.csv")

OUT_CSV = Path("analysis/prepost_reconstitution_summary.csv")
OUT_SVG = Path("analysis/prepost_reconstitution_check.svg")

DEP_VARS = ["amihud_illiq", "volatility", "price_info"]

CUTOFF = 2000.0
BANDWIDTH = 50.0
TAUS = list(range(-2, 3))
PRE_TAUS = [-2, -1]
POST_TAUS = [0, 1, 2]

MONTH_TO_POS = {3: 0, 6: 1, 9: 2, 12: 3}


def parse_date(s):
    return datetime.strptime(s, "%Y-%m-%d")


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def load_rank_map():
    out = {}
    with RANK_PATH.open("r", newline="") as f:
        rd = csv.DictReader(f)
        for r in rd:
            cusip = r.get("cusip", "").strip()
            ds = r.get("date", "").strip()
            rv = piv.parse_float(r.get("Rank", ""))
            if not cusip or not ds or rv is None:
                continue
            yr = parse_date(ds).year
            k = (cusip, yr)
            if k in out:
                if rv < out[k]:
                    out[k] = rv
            else:
                out[k] = rv
    return out


def load_main_lookup():
    # (cusip, abs_q) -> tuple(dep vars)
    out = {}
    with MAIN_PATH.open("r", newline="") as f:
        rd = csv.DictReader(f)
        for r in rd:
            cusip = r.get("cusip", "").strip()
            ds = r.get("date", "").strip()
            if not cusip or not ds:
                continue
            dt = parse_date(ds)
            pos = MONTH_TO_POS.get(dt.month)
            if pos is None:
                continue
            abs_q = dt.year * 4 + pos
            vals = []
            ok = True
            for d in DEP_VARS:
                v = piv.parse_float(r.get(d, ""))
                if v is None:
                    ok = False
                    break
                vals.append(v)
            if ok:
                out[(cusip, abs_q)] = tuple(vals)
    return out


def collect(rank_map, lookup):
    # agg[(dep, group, tau)] -> [sum, n]
    agg = {}
    for d in DEP_VARS:
        for g in ("control", "treated", "all"):
            for t in TAUS:
                agg[(d, g, t)] = [0.0, 0]

    for (cusip, year), rank in rank_map.items():
        if abs(rank - CUTOFF) > BANDWIDTH:
            continue
        treated = rank <= CUTOFF
        group = "treated" if treated else "control"

        # event time 0 = June of event year
        event0_absq = year * 4 + 1

        for tau in TAUS:
            abs_q = event0_absq + tau
            rec = lookup.get((cusip, abs_q))
            if rec is None:
                continue
            for i, d in enumerate(DEP_VARS):
                v = rec[i]
                a = agg[(d, group, tau)]
                a[0] += v
                a[1] += 1
                b = agg[(d, "all", tau)]
                b[0] += v
                b[1] += 1

    rows = []
    for d in DEP_VARS:
        for g in ("control", "treated", "all"):
            for t in TAUS:
                s, n = agg[(d, g, t)]
                m = (s / n) if n > 0 else float("nan")
                rows.append(
                    {
                        "dep_var": d,
                        "group": g,
                        "tau": t,
                        "mean": m,
                        "n": n,
                    }
                )
    return rows


def mean_over(rows, dep, grp, taus):
    vals = []
    for t in taus:
        r = next((x for x in rows if x["dep_var"] == dep and x["group"] == grp and x["tau"] == t), None)
        if r and np.isfinite(r["mean"]):
            vals.append(r["mean"])
    return float(np.mean(vals)) if vals else float("nan")


def render(rows):
    W, H = 1320, 500
    ML, MR, MT, MB = 72, 35, 62, 78
    GAP = 34
    pw = (W - ML - MR - GAP * 2) / 3
    ph = H - MT - MB

    vals = [r["mean"] for r in rows if np.isfinite(r["mean"]) and r["group"] in ("control", "treated")]
    ymin, ymax = (min(vals), max(vals)) if vals else (-1.0, 1.0)
    if ymax <= ymin:
        ymax = ymin + 1.0
    pad = 0.12 * (ymax - ymin)
    ymin -= pad
    ymax += pad

    pos = {t: i for i, t in enumerate(TAUS)}

    def xpx(t, left):
        return left + (pos[t] / (len(TAUS) - 1)) * pw

    def ypx(v):
        return MT + (ymax - v) / (ymax - ymin) * ph

    parts = []
    parts.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">')
    parts.append("<style>")
    parts.append("text{font-family:Arial,Helvetica,sans-serif;fill:#111827}")
    parts.append(".title{font-size:19px;font-weight:700}")
    parts.append(".axis{stroke:#374151;stroke-width:1.2}")
    parts.append(".grid{stroke:#e5e7eb;stroke-width:1}")
    parts.append(".ct{stroke:#2563eb;stroke-width:2.2;fill:none}")
    parts.append(".tr{stroke:#dc2626;stroke-width:2.2;fill:none}")
    parts.append(".ptct{fill:#1d4ed8}")
    parts.append(".pttr{fill:#b91c1c}")
    parts.append(".shade{fill:#f3f4f6}")
    parts.append(".lbl{font-size:13px}")
    parts.append(".tick{font-size:11px;fill:#4b5563}")
    parts.append(".chg{font-size:11px;fill:#374151}")
    parts.append("</style>")
    parts.append(f'<rect x="0" y="0" width="{W}" height="{H}" fill="white"/>')
    parts.append(f'<text x="{W/2}" y="34" text-anchor="middle" class="title">Fine-Grained Pre/Post Check Around Russell 2000 Reconstitution</text>')
    parts.append(f'<line x1="{W-285}" y1="26" x2="{W-260}" y2="26" class="ct"/>')
    parts.append(f'<text x="{W-253}" y="30" class="tick">Control (&gt;2000)</text>')
    parts.append(f'<line x1="{W-160}" y1="26" x2="{W-135}" y2="26" class="tr"/>')
    parts.append(f'<text x="{W-128}" y="30" class="tick">Treated (≤2000)</text>')

    for i, dep in enumerate(DEP_VARS):
        left = ML + i * (pw + GAP)
        right = left + pw

        # shade post region (tau >= 0)
        x0 = xpx(0, left)
        parts.append(f'<rect x="{x0:.2f}" y="{MT:.2f}" width="{right-x0:.2f}" height="{ph:.2f}" class="shade"/>')
        parts.append(f'<rect x="{left:.2f}" y="{MT:.2f}" width="{pw:.2f}" height="{ph:.2f}" fill="none" class="axis"/>')
        parts.append(f'<text x="{left + pw/2:.2f}" y="{MT-12:.2f}" text-anchor="middle" class="lbl">{esc(dep)}</text>')

        for frac in [0.0, 0.5, 1.0]:
            yv = ymin + frac * (ymax - ymin)
            yp = ypx(yv)
            parts.append(f'<line x1="{left:.2f}" y1="{yp:.2f}" x2="{right:.2f}" y2="{yp:.2f}" class="grid"/>')

        for grp, line_cls, pt_cls in [("control", "ct", "ptct"), ("treated", "tr", "pttr")]:
            pts = []
            for t in TAUS:
                r = next((x for x in rows if x["dep_var"] == dep and x["group"] == grp and x["tau"] == t), None)
                if r is None or (not np.isfinite(r["mean"])):
                    continue
                pts.append((xpx(t, left), ypx(r["mean"])))
            if len(pts) >= 2:
                d = "M " + " L ".join(f"{x:.2f} {y:.2f}" for x, y in pts)
                parts.append(f'<path d="{d}" class="{line_cls}"/>')
            for x, y in pts:
                parts.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="2.9" class="{pt_cls}"/>')

        # pre/post changes
        pre_c = mean_over(rows, dep, "control", PRE_TAUS)
        post_c = mean_over(rows, dep, "control", POST_TAUS)
        pre_t = mean_over(rows, dep, "treated", PRE_TAUS)
        post_t = mean_over(rows, dep, "treated", POST_TAUS)

        dc = post_c - pre_c if np.isfinite(pre_c) and np.isfinite(post_c) else float("nan")
        dt = post_t - pre_t if np.isfinite(pre_t) and np.isfinite(post_t) else float("nan")
        dd = dt - dc if np.isfinite(dc) and np.isfinite(dt) else float("nan")

        if np.isfinite(dc):
            parts.append(f'<text x="{left+8:.2f}" y="{MT+16:.2f}" class="chg">Δ control (post-pre): {dc:.4f}</text>')
        if np.isfinite(dt):
            parts.append(f'<text x="{left+8:.2f}" y="{MT+30:.2f}" class="chg">Δ treated (post-pre): {dt:.4f}</text>')
        if np.isfinite(dd):
            parts.append(f'<text x="{left+8:.2f}" y="{MT+44:.2f}" class="chg">Δ gap: {dd:.4f}</text>')

        for t in TAUS:
            x = xpx(t, left)
            parts.append(f'<line x1="{x:.2f}" y1="{MT+ph:.2f}" x2="{x:.2f}" y2="{MT+ph+4:.2f}" class="axis"/>')
            parts.append(f'<text x="{x:.2f}" y="{MT+ph+18:.2f}" text-anchor="middle" class="tick">{t}</text>')

    parts.append(f'<text x="{W/2:.2f}" y="{H-16:.2f}" text-anchor="middle" class="lbl">Event time tau (quarters): 0 = June reconstitution quarter; shaded area is post period. Local sample: |Rank-2000| ≤ 50</text>')
    parts.append("</svg>")

    OUT_SVG.write_text("\n".join(parts), encoding="utf-8")


def main():
    rank_map = load_rank_map()
    lookup = load_main_lookup()
    rows = collect(rank_map, lookup)

    with OUT_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["dep_var", "group", "tau", "mean", "n"])
        w.writeheader()
        for r in rows:
            w.writerow(r)

    render(rows)
    print(f"Saved: {OUT_CSV}")
    print(f"Saved: {OUT_SVG}")


if __name__ == "__main__":
    main()
