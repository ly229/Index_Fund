# How the Market Score Is Calculated

The `market score` in the nonlinear analysis is constructed in four steps:

1. **Estimate quadratic FE models for each market-quality outcome**
   - Model: `y = b1*ind_own + b2*ind_own^2 + controls + firm FE + time FE`
   - Outcomes:
     - `amihud_illiq`
     - `volatility`
     - `price_info`

2. **Build predicted change curves over an `ind_own` grid**
   - For each outcome: `g(x) = b1*x + b2*x^2`
   - Convert to change relative to a reference level (`x_ref`, median `ind_own`):
     - `delta(x) = g(x) - g(x_ref)`

3. **Standardize each outcome curve (z-score over the grid)**
   - This makes all components comparable in units:
     - `z_amihud(x)`, `z_volatility(x)`, `z_price_info(x)`

4. **Combine with economic signs for market quality**
   - Lower illiquidity is better, lower volatility is better, higher price informativeness is better.
   - Composite score:
     - `score(x) = -z_amihud(x) - z_volatility(x) + z_price_info(x)`

## Interpretation of the score
- Higher `score(x)` = better overall market quality under this composite criterion.
- **Optimal indexing level** = `x` that maximizes `score(x)`.
- **Good indexing range** = top 5% of `score(x)` values across the evaluated `ind_own` grid.

## Implementation reference
- Script: `analysis/nonlinearity_indexing_panel.py`
- Summary output: `analysis/nonlinearity/nonlinear_indexing_summary.md`
- Score grid output: `analysis/nonlinearity/nonlinear_indexing_score_grid.csv`
