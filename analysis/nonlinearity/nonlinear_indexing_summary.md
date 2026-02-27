# Nonlinearity: How Much Indexing Is Good?

Model: panel FE with quadratic term
- `y_it = b1*ind_own + b2*ind_own^2 + controls + firm FE + time FE`
- SE: two-way clustered (firm, date)

## Core answer
- Composite optimum indexing level: `ind_own ≈ 0.0379`
- High-quality range (top 5% score): `[0.0328, 0.0430]`

## Quadratic terms
- `amihud_illiq`: b1=0.01071 (p=0.748), b2=0.1882 (p=0.0469), turning=-0.0284, in_support=False
- `volatility`: b1=-0.325 (p=2.12e-07), b2=0.9991 (p=1.49e-06), turning=0.1626, in_support=True
- `price_info`: b1=-10.36 (p=0), b2=26.7 (p=3.46e-08), turning=0.1939, in_support=True

Interpretation note: lower `amihud_illiq` and `volatility` are better; higher `price_info` is better.
