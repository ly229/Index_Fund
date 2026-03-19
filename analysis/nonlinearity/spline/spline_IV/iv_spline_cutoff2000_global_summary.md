# IV Spline Nonlinearity Results

Specification:
- 2SLS with endogenous natural cubic spline basis in `ind_own`
- Preferred IV design: global sample, cutoff=2000, bandwidth=None
- Excluded instruments: `z_cutoff * running^k`, `k=0..4`
- Controls: `c_firm_size`, `c_dollar_vol`, `running^k`, `k=1..4`
- FE: firm + date (via two-way demeaning)
- SE: firm-clustered
- Spline knots in raw `ind_own`: `0.0265, 0.0510, 0.0952`

## Core answer
- Composite optimum indexing level: `ind_own ≈ 0.1579`
- High-quality range (top 5% score): `[0.1525, 0.1628]`

## Outcome summaries
- `amihud_illiq`: best `ind_own ≈ 0.2267`, min first-stage F across spline terms = 15.74, n = 235544
- `volatility`: best `ind_own ≈ 0.0327`, min first-stage F across spline terms = 15.91, n = 234897
- `price_info`: best `ind_own ≈ 0.1470`, min first-stage F across spline terms = 15.66, n = 234281

Interpretation note: this is the causal spline extension of the rank-cutoff IV design. If first-stage strength is weak for some spline terms, interpret the curve shape cautiously.
