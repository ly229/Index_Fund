# Nonlinear IV (Quadratic) Results

Specification:
- 2SLS with endogenous: `ind_own`, `ind_own^2`
- Excluded instruments: `z*running^k, k=0..2`
- Controls: `c_firm_size`, `c_dollar_vol`, `running^k, k=1..2`
- FE: firm + date (via two-way demeaning)
- Local sample: cutoff=1000, bandwidth=150

## amihud_illiq
- b1(ind_own)=0.1407 (p=0.305), b2(ind_own^2)=-0.7455 (p=0.261)
- turning point=0.0944, in_support=True [q01=0.0104, q99=0.1613]
- first-stage joint F: ind_own=1.458 (p=0.224); ind_own^2=1.112 (p=0.343)

## volatility
- b1(ind_own)=5.406 (p=0.486), b2(ind_own^2)=-26.38 (p=0.468)
- turning point=0.1025, in_support=True [q01=0.0104, q99=0.1613]
- first-stage joint F: ind_own=1.455 (p=0.225); ind_own^2=1.112 (p=0.342)

## price_info
- b1(ind_own)=114 (p=0.415), b2(ind_own^2)=-491.3 (p=0.452)
- turning point=0.1160, in_support=True [q01=0.0104, q99=0.1613]
- first-stage joint F: ind_own=1.516 (p=0.208); ind_own^2=1.144 (p=0.33)
