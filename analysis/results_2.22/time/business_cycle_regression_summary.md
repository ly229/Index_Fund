# Business Cycle Regression Summary

- Spec: two-way FE (cusip + date), controls: c_firm_size, c_dollar_vol
- Recession dummy defined for: 2001Q1–Q4, 2007Q4–2009Q2, 2019Q4–2020Q2
- Model: y = ind_own + recession + ind_own×recession + controls + FE
- SEs: two-way clustered (cusip × date)

| Dependent | N | beta(ind_own) | t | beta(recession) | t | beta(ind_own×recession) | t | within R² |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| amihud_illiq | 239287 | 0.0853563 | 3.682 | -21.3804 | -0.000 | -0.21454 | -2.534 | 0.0916 |
| volatility | 239287 | -0.0743259 | -3.258 | -109.45 | -0.000 | 0.141461 | 1.351 | 0.2217 |
| price_info | 239287 | -3.26549 | -13.798 | -373.559 | -0.000 | -0.526318 | -0.673 | 0.0297 |