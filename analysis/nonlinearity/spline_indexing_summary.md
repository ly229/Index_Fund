# Spline Nonlinearity: How Much Indexing Is Good?

Model: panel FE with natural cubic spline in `ind_own`
- `y_it = f(ind_own_it) + controls + firm FE + time FE`
- `f(.)`: natural cubic spline with interior knots at the full-sample `ind_own` quantiles 25%, 50%, 75%
- Knot values: `0.0265, 0.0510, 0.0952`
- SE: two-way clustered (firm, date)

## Core answer
- Composite optimum indexing level: `ind_own ≈ 0.0548`
- High-quality range (top 5% score): `[0.0500, 0.0603]`

## Outcome-specific optima
- `amihud_illiq`: best `ind_own ≈ 0.0467`, within R^2 = 0.0898, n = 241199
- `volatility`: best `ind_own ≈ 0.0733`, within R^2 = 0.2200, n = 240515
- `price_info`: best `ind_own ≈ 0.0104`, within R^2 = 0.0132, n = 239758

Interpretation note: lower `amihud_illiq` and `volatility` are better; higher `price_info` is better.
This spline specification is descriptive non-IV evidence. It is more flexible than the quadratic benchmark and is useful for checking whether the implied optimum is driven by functional-form restrictions.
