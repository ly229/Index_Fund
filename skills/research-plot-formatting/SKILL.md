---
name: research-plot-formatting
description: Reformat existing analysis plots into finance/economics research style. Use when improving Matplotlib-generated figures for papers, theses, appendices, or presentations, especially when the goal is reproducible script-level styling rather than manual editing.
---

# Research Plot Formatting

Use this skill when a user wants existing plots upgraded to research standard.

## Goal

Convert functional plots into publication-style figures by editing the generating script, rerunning it, and preserving the empirical content.

Default priorities:
- improve readability before decoration
- keep style changes reproducible in code
- preserve the underlying estimates and interpretation
- match the visual language across related figures

## Workflow

1. Find the generating script before editing the figure.
2. Read only the plotting block and the output paths first.
3. Identify the figure’s role:
   - coefficient plot
   - fitted-line plot
   - multi-panel comparison
   - score/optimization plot
4. Revise layout before color:
   - title hierarchy
   - panel arrangement
   - legend placement
   - axis labels and scale formatting
   - whitespace balance
5. Add research-style polish:
   - serif typography
   - light grids
   - muted spines
   - consistent line widths
   - restrained color palette
6. Add annotations only if they carry substantive meaning:
   - optimum
   - turning point
   - support range
   - knot locations
   - confidence intervals
7. Regenerate the figure from the script.
8. Check whether the export succeeded and whether layout changes created new crowding or dead space.

## Standard Moves

### Typography and Axis Style

- Prefer serif fonts such as `STIX Two Text`, `Times New Roman`, or `DejaVu Serif`.
- Remove top and right spines.
- Use light horizontal grid lines only.
- Keep labels explicit and short.
- Format shares as percentages when the x-axis is ownership or exposure.

### Titles

- Use one short main title.
- Remove explanatory subtitle text unless it adds real information.
- If multiple panels are present, panel titles should be short and consistent.
- Add `a)`, `b)`, `c)` markers only when they help cross-reference the text.

### Legends

- Prefer one shared legend for multi-panel figures.
- Move the legend outside the data region when it competes with the curves.
- Bottom-of-figure legends usually work better than repeated panel legends.

### Panel Layout

- Keep comparable panels the same size.
- Use a single row when width is available and panels are simple.
- Use a `2 + 1` triangular layout when the figure looks too wide or generic, but keep the third panel the same size as the others.
- Avoid stretching one panel across a full row unless it represents a different object.

### Data Layer Additions

Use only if empirically justified:
- confidence bands
- shaded support range
- vertical lines for knots or turning points
- labeled optimum markers

Avoid adding explanatory text like “lower is better” inside panels unless the user explicitly wants teaching-style figures.

## Notes by Figure Type

### Fitted-Line Plots

- Emphasize the curve shape.
- Mark turning points only when they are inside support.
- Shade support lightly, not heavily.

### Spline Multi-Panel Plots

- Show 95% confidence bands if available.
- Mark interior knots with thin dashed vertical lines.
- If optima are important, label them directly on each panel.
- Keep panel titles and legend from competing with the data.

### Score or “Good Range” Plots

- Highlight the top range with a light band.
- Mark the optimum with a vertical line and point annotation.
- Use a compact note box for the best range when helpful.

## Editing Rules

- Edit the source plotting script, not the exported SVG directly.
- Reuse any existing style helpers if they exist.
- Prefer incremental layout edits over rewriting the whole script.
- If the user asks for repeated refinement, change only the requested elements.

## Verification

After rerunning:
- confirm the output path was written
- check that titles, legend, and panel sizes match the request
- mention any estimation warnings separately from plotting success

## Typical Summary Back to User

Keep the close-out short:
- what visual changes were made
- which file was regenerated
- whether underlying estimation warnings still remain
