# Quadratic IV Robustness Grid

Ranking criterion: maximize `min(F_ind_own, F_ind_own_sq)` within each dependent variable.

## amihud_illiq
- Best spec: cutoff=1500, bandwidth=200, min first-stage F=3.007
- Coefficients: b1=-8.228 (p=0.945), b2=31.08 (p=0.946)

## volatility
- Best spec: cutoff=1500, bandwidth=200, min first-stage F=3.034
- Coefficients: b1=98.25 (p=0.949), b2=-382.1 (p=0.948)

## price_info
- Best spec: cutoff=1500, bandwidth=200, min first-stage F=3.058
- Coefficients: b1=154.1 (p=0.96), b2=-537.2 (p=0.963)

Note: If best min first-stage F remains far below 10, nonlinear IV remains weakly identified.