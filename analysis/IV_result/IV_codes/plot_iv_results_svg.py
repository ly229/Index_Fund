#!/usr/bin/env python3
import csv
from pathlib import Path

IN_PATH = Path('analysis/iv_panel_results_rank_cutoff.csv')
OUT_PATH = Path('analysis/iv_rank_cutoff_coefficients.svg')

rows = []
with IN_PATH.open() as f:
    reader = csv.DictReader(f)
    for r in reader:
        beta = float(r['beta_endog'])
        se = float(r['se_endog'])
        lo = beta - 1.96 * se
        hi = beta + 1.96 * se
        rows.append({
            'dep_var': r['dep_var'],
            'beta': beta,
            'se': se,
            'lo': lo,
            'hi': hi,
        })

if not rows:
    raise SystemExit('No rows in results CSV')

# Layout
W, H = 1100, 500
ML, MR, MT, MB = 270, 80, 50, 70
plot_w = W - ML - MR
plot_h = H - MT - MB

xmin = min(r['lo'] for r in rows)
xmax = max(r['hi'] for r in rows)
pad = 0.06 * (xmax - xmin if xmax > xmin else 1.0)
xmin -= pad
xmax += pad


def x_to_px(x):
    return ML + (x - xmin) / (xmax - xmin) * plot_w


def esc(s):
    return (s.replace('&', '&amp;')
              .replace('<', '&lt;')
              .replace('>', '&gt;'))

# Vertical positions (top to bottom in CSV order)
y_step = plot_h / max(len(rows), 1)
for i, r in enumerate(rows):
    r['y'] = MT + (i + 0.5) * y_step

parts = []
parts.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">')
parts.append('<style>')
parts.append('text{font-family:Arial,Helvetica,sans-serif;fill:#1f2937}')
parts.append('.title{font-size:22px;font-weight:700}')
parts.append('.axis{stroke:#374151;stroke-width:1.2}')
parts.append('.grid{stroke:#e5e7eb;stroke-width:1}')
parts.append('.ci{stroke:#2563eb;stroke-width:2.6}')
parts.append('.pt{fill:#1d4ed8}')
parts.append('.lbl{font-size:14px}')
parts.append('.tick{font-size:12px;fill:#4b5563}')
parts.append('</style>')

parts.append(f'<rect x="0" y="0" width="{W}" height="{H}" fill="white"/>')
parts.append(f'<text x="{W/2}" y="30" text-anchor="middle" class="title">IV Effect of ind_own (95% CI)</text>')

# Grid/ticks
ntick = 6
for i in range(ntick + 1):
    xv = xmin + (xmax - xmin) * i / ntick
    xp = x_to_px(xv)
    parts.append(f'<line x1="{xp:.2f}" y1="{MT}" x2="{xp:.2f}" y2="{H-MB}" class="grid"/>')
    parts.append(f'<text x="{xp:.2f}" y="{H-MB+22}" text-anchor="middle" class="tick">{xv:.2f}</text>')

# zero line
if xmin < 0 < xmax:
    x0 = x_to_px(0.0)
    parts.append(f'<line x1="{x0:.2f}" y1="{MT}" x2="{x0:.2f}" y2="{H-MB}" stroke="#111827" stroke-width="1.5" stroke-dasharray="5 4"/>')
    parts.append(f'<text x="{x0+6:.2f}" y="{MT+14}" class="tick">0</text>')

# Axes border
parts.append(f'<line x1="{ML}" y1="{MT}" x2="{ML}" y2="{H-MB}" class="axis"/>')
parts.append(f'<line x1="{ML}" y1="{H-MB}" x2="{W-MR}" y2="{H-MB}" class="axis"/>')

for r in rows:
    y = r['y']
    x1 = x_to_px(r['lo'])
    x2 = x_to_px(r['hi'])
    xb = x_to_px(r['beta'])

    parts.append(f'<line x1="{x1:.2f}" y1="{y:.2f}" x2="{x2:.2f}" y2="{y:.2f}" class="ci"/>')
    parts.append(f'<line x1="{x1:.2f}" y1="{y-6:.2f}" x2="{x1:.2f}" y2="{y+6:.2f}" class="ci"/>')
    parts.append(f'<line x1="{x2:.2f}" y1="{y-6:.2f}" x2="{x2:.2f}" y2="{y+6:.2f}" class="ci"/>')
    parts.append(f'<circle cx="{xb:.2f}" cy="{y:.2f}" r="5.2" class="pt"/>')

    label = esc(r['dep_var'])
    val = f"{r['beta']:.3f} [{r['lo']:.3f}, {r['hi']:.3f}]"
    parts.append(f'<text x="{ML-14}" y="{y+5:.2f}" text-anchor="end" class="lbl">{label}</text>')
    parts.append(f'<text x="{W-MR+8}" y="{y+5:.2f}" class="tick">{esc(val)}</text>')

parts.append(f'<text x="{(ML + (W-MR))/2:.2f}" y="{H-18}" text-anchor="middle" class="lbl">Coefficient on ind_own</text>')
parts.append('</svg>')

OUT_PATH.write_text('\n'.join(parts), encoding='utf-8')
print(f'Wrote {OUT_PATH}')
