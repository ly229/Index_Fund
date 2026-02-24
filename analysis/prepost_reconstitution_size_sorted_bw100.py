#!/usr/bin/env python3
import csv
from datetime import datetime
from pathlib import Path

import numpy as np

import panel_iv_rank_cutoff as piv

MAIN_PATH = Path("data/df_regression.csv")
RANK_PATH = Path("data/Russell_3_rank.csv")

OUT_CSV = Path("analysis/prepost_reconstitution_size_sorted_bw100.csv")
OUT_SVG = Path("analysis/prepost_reconstitution_size_sorted_bw100.svg")

DEP_VARS = ["amihud_illiq", "volatility", "price_info"]
SIZE_BUCKETS = ["Small", "Mid", "Large"]
TAUS = [-2, -1, 0, 1, 2]

CUTOFF = 2000.0
BANDWIDTH = 100.0

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
            y = parse_date(ds).year
            k = (cusip, y)
            if k in out:
                if rv < out[k]:
                    out[k] = rv
            else:
                out[k] = rv
    return out


def load_main_lookup():
    # (cusip, abs_q) -> (dep1, dep2, dep3, mktcap)
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
            mktcap = piv.parse_float(r.get("mktcap", ""))
            if (not ok) or (mktcap is None):
                continue
            out[(cusip, abs_q)] = (vals[0], vals[1], vals[2], mktcap)
    return out


def assign_size_bucket(rank_map, lookup):
    # assign by year using pre-reconstitution size at tau=-1 (March).
    # output key: (cusip, year) -> bucket
    per_year = {}
    for (cusip, year), rank in rank_map.items():
        if abs(rank - CUTOFF) > BANDWIDTH:
            continue
        event0_absq = year * 4 + 1
        rec = lookup.get((cusip, event0_absq - 1))
        if rec is None:
            continue
        mktcap = rec[3]
        if mktcap <= 0:
            continue
        per_year.setdefault(year, []).append((cusip, float(mktcap)))

    out = {}
    for year, arr in per_year.items():
        if len(arr) < 3:
            continue
        vals = np.array([x[1] for x in arr], dtype=float)
        q1 = float(np.quantile(vals, 1 / 3))
        q2 = float(np.quantile(vals, 2 / 3))
        for cusip, v in arr:
            if v <= q1:
                b = "Small"
            elif v <= q2:
                b = "Mid"
            else:
                b = "Large"
            out[(cusip, year)] = b
    return out


def collect(rank_map, lookup, size_map):
    agg = {}
    for d in DEP_VARS:
        for b in SIZE_BUCKETS:
            for t in TAUS:
                agg[(d, b, t)] = [0.0, 0]

    for (cusip, year), rank in rank_map.items():
        if abs(rank - CUTOFF) > BANDWIDTH:
            continue
        bucket = size_map.get((cusip, year))
        if bucket is None:
            continue

        event0_absq = year * 4 + 1
        for tau in TAUS:
            rec = lookup.get((cusip, event0_absq + tau))
            if rec is None:
                continue
            for i, d in enumerate(DEP_VARS):
                v = rec[i]
                a = agg[(d, bucket, tau)]
                a[0] += v
                a[1] += 1

    rows = []
    for d in DEP_VARS:
        for b in SIZE_BUCKETS:
            for t in TAUS:
                s, n = agg[(d, b, t)]
                m = (s / n) if n > 0 else float("nan")
                rows.append({"dep_var": d, "size_bucket": b, "tau": t, "mean": m, "n": n})
    return rows


def render(rows):
    W, H = 1320, 500
    ML, MR, MT, MB = 72, 35, 62, 78
    GAP = 34
    pw = (W - ML - MR - GAP * 2) / 3
    ph = H - MT - MB

    vals = [r["mean"] for r in rows if np.isfinite(r["mean"])]
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

    line_style = {
        "Small": ("#2563eb", "#1d4ed8"),
        "Mid": ("#dc2626", "#b91c1c"),
        "Large": ("#059669", "#047857"),
    }

    parts = []
    parts.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">')
    parts.append("<style>")
    parts.append("text{font-family:Arial,Helvetica,sans-serif;fill:#111827}")
    parts.append(".title{font-size:19px;font-weight:700}")
    parts.append(".axis{stroke:#374151;stroke-width:1.2}")
    parts.append(".grid{stroke:#e5e7eb;stroke-width:1}")
    parts.append(".shade{fill:#f3f4f6}")
    parts.append(".lbl{font-size:13px}")
    parts.append(".tick{font-size:11px;fill:#4b5563}")
    parts.append("</style>")
    parts.append(f'<rect x="0" y="0" width="{W}" height="{H}" fill="white"/>')
    parts.append(f'<text x="{W/2}" y="34" text-anchor="middle" class="title">Size-Sorted Pre/Post Around Russell 2000 Reconstitution (|Rank-2000| ≤ 100)</text>')

    for i, dep in enumerate(DEP_VARS):
        left = ML + i * (pw + GAP)
        right = left + pw

        x0 = xpx(0, left)
        parts.append(f'<rect x="{x0:.2f}" y="{MT:.2f}" width="{right-x0:.2f}" height="{ph:.2f}" class="shade"/>')
        parts.append(f'<rect x="{left:.2f}" y="{MT:.2f}" width="{pw:.2f}" height="{ph:.2f}" fill="none" class="axis"/>')
        parts.append(f'<text x="{left + pw/2:.2f}" y="{MT-12:.2f}" text-anchor="middle" class="lbl">{esc(dep)}</text>')

        for frac in [0.0, 0.5, 1.0]:
            yv = ymin + frac * (ymax - ymin)
            yp = ypx(yv)
            parts.append(f'<line x1="{left:.2f}" y1="{yp:.2f}" x2="{right:.2f}" y2="{yp:.2f}" class="grid"/>')

        for b in SIZE_BUCKETS:
            line_c, pt_c = line_style[b]
            pts = []
            for t in TAUS:
                r = next((x for x in rows if x["dep_var"] == dep and x["size_bucket"] == b and x["tau"] == t), None)
                if r is None or (not np.isfinite(r["mean"])):
                    continue
                pts.append((xpx(t, left), ypx(r["mean"])))
            if len(pts) >= 2:
                d = "M " + " L ".join(f"{x:.2f} {y:.2f}" for x, y in pts)
                parts.append(f'<path d="{d}" stroke="{line_c}" stroke-width="2.2" fill="none"/>')
            for x, y in pts:
                parts.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="2.9" fill="{pt_c}"/>')

        for t in TAUS:
            x = xpx(t, left)
            parts.append(f'<line x1="{x:.2f}" y1="{MT+ph:.2f}" x2="{x:.2f}" y2="{MT+ph+4:.2f}" class="axis"/>')
            parts.append(f'<text x="{x:.2f}" y="{MT+ph+18:.2f}" text-anchor="middle" class="tick">{t}</text>')

    parts.append(f'<text x="{W/2:.2f}" y="{H-32:.2f}" text-anchor="middle" class="lbl">Event time tau in quarters (0=June reconstitution); size buckets formed by year using pre-reconstitution (tau=-1) market cap terciles</text>')
    # Legend under the graph.
    lx = W / 2 - 170
    ly = H - 14
    parts.append(f'<line x1="{lx:.2f}" y1="{ly:.2f}" x2="{lx+24:.2f}" y2="{ly:.2f}" stroke="#2563eb" stroke-width="2.2"/>')
    parts.append(f'<text x="{lx+30:.2f}" y="{ly+4:.2f}" class="tick">Small</text>')
    parts.append(f'<line x1="{lx+95:.2f}" y1="{ly:.2f}" x2="{lx+119:.2f}" y2="{ly:.2f}" stroke="#dc2626" stroke-width="2.2"/>')
    parts.append(f'<text x="{lx+125:.2f}" y="{ly+4:.2f}" class="tick">Mid</text>')
    parts.append(f'<line x1="{lx+180:.2f}" y1="{ly:.2f}" x2="{lx+204:.2f}" y2="{ly:.2f}" stroke="#059669" stroke-width="2.2"/>')
    parts.append(f'<text x="{lx+210:.2f}" y="{ly+4:.2f}" class="tick">Large</text>')
    parts.append("</svg>")

    OUT_SVG.write_text("\n".join(parts), encoding="utf-8")


def main():
    rank_map = load_rank_map()
    lookup = load_main_lookup()
    size_map = assign_size_bucket(rank_map, lookup)
    rows = collect(rank_map, lookup, size_map)

    with OUT_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["dep_var", "size_bucket", "tau", "mean", "n"])
        w.writeheader()
        for r in rows:
            w.writerow(r)

    render(rows)
    print(f"Saved: {OUT_CSV}")
    print(f"Saved: {OUT_SVG}")


if __name__ == "__main__":
    main()
