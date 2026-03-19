---
name: regression-table-formatting
description: Convert regression-result CSV files into publication-ready LaTeX tables with booktabs rules, multi-line coefficient cells, significance stars, and a full-width notes block. Use when the user asks to format econometrics or finance regression output for papers, theses, or appendices.
---

# Regression Table Formatting

## Overview
Turn regression CSV output into a clean LaTeX table suitable for publication. Default to `booktabs` styling with `\toprule`, `\midrule`, and `\bottomrule`, compact spacing, significance stars, and a notes block that spans the full table width.

## Use This Skill When
- The input is a CSV or similar flat file with coefficients, standard errors, t-stats, sample sizes, or fit statistics.
- The user wants a publication-ready table rather than a raw data export.
- The table should follow economics/finance conventions: clear panel structure, coefficient formatting, and footnotes below the full table.

## Core Workflow
1. Inspect the CSV schema and identify:
   - dependent-variable groups or panels,
   - row variables,
   - coefficient / standard error / t-stat columns,
   - sample size and fit statistics.
2. Build a deterministic row order and label map.
3. Format coefficients with significance stars derived from t-statistics.
4. Render each cell as a two-line entry when the user wants standard errors underneath coefficients.
   - Preferred LaTeX pattern: `\shortstack{coef\\(se)}`
5. Use `booktabs` rules:
   - `\toprule` after the header,
   - `\midrule` between sections and before summary rows,
   - `\bottomrule` at the end.
6. Put notes in a full-width block beneath the table body.
   - Prefer `\parbox{\textwidth}{...}` or an equivalent full-width construct.
7. Regenerate the `.tex` file and verify the emitted output, not just the script.

## Formatting Defaults
- Use compact column spacing when tables are wide.
- Keep variable names short and reader-friendly.
- Preserve exact numeric values unless the user asks for rounding changes.
- Report observations and fit statistics at the bottom of each panel or table.
- Use a plain note sentence for significance thresholds, sample construction, and any caveats.

## Recommended LaTeX Pattern
- Table body:
  - header row
  - `\midrule`
  - coefficient rows
  - summary rows
  - `\bottomrule`
- Notes block:
  - place after the table body and before `\end{threeparttable}` if used
  - span the full width of the table
  - keep it visually separate from panel content

## Verification
- Confirm the output contains `\toprule`, `\midrule`, and `\bottomrule`.
- Confirm coefficient cells render as two lines when requested.
- Confirm notes are not trapped inside a narrow subtable.
- Confirm the generated `.tex` compiles cleanly in the target document.

## Reference Implementation
If a repo already contains a table builder script, update that script rather than hand-editing the output `.tex` file. Keep the generator as the source of truth.

