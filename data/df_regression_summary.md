# `df_regression.csv` summary

## Snapshot
- 799,337 quarterly firm-quarter observations from 1993-03-31 through 2023-12-31 (124 quarter-ends) across 26,395 distinct CUSIPs.
- Each row is keyed by `cusip` + `date` and pairs the dependent variable (`bas`) with firm controls covering size, trading activity, liquidity, and ownership.
- Missing values exist only in `volatility`, `price_info`, `ind_own`, and `r_ins_own`; the following statistics are all computed on the available (non-missing and non-infinite) cells for each column.

## Variable descriptions
- `bas`: primary dependent variable (bid-ask spread or related transaction-cost measure); strongly right-skewed with a few large values.
- `mktcap`, `c_firm_size`, `c_dollar_vol`, `r_tno`: standard size/volume controls capturing market capitalization, firm size, dollar trading volume, and turnover.
- `amihud_illiq`: Amihud-style illiquidity measure (ratio of absolute return to dollar volume).
- `volatility`: realized volatility estimate (about 0.18 on average; missing for 5,599 obs).
- `price_info`: price-information proxy; distribution spans from about –6.9 to +6.9 (negative values occur when the numerator and denominator have opposing signs), with 19,814 missing values.
- `ind_own`: industry ownership share (available for ~241K obs).
- `r_ins_own`: institutional ownership ratio (available for ~498K obs).

## Summary statistics (non-missing values only)
| Variable | Count | Missing | Mean | Std dev | Min | 25% | Median | 75% | Max |
|---|---|---|---|---|---|---|---|---|---|
| `bas` | 799,337 | 0 | 0.2543 | 4.6849 | –30.0000 | 0.0200 | 0.0625 | 0.2500 | 1,428.8999 |
| `mktcap` | 799,337 | 0 | 3,569.0351 | 23,882.4701 | 0.0101 | 89.8660 | 322.8970 | 1,344.8743 | 2,977,574.3704 |
| `c_firm_size` | 799,337 | 0 | 5.8858 | 2.0545 | –4.5952 | 4.4983 | 5.7773 | 7.2041 | 14.9066 |
| `c_dollar_vol` | 799,337 | 0 | 18.3176 | 2.7162 | 9.2103 | 16.4438 | 18.2980 | 20.2703 | 28.7857 |
| `r_tno` | 799,337 | 0 | 0.7287 | 7.0680 | 0.0000 | 0.1177 | 0.2818 | 0.5842 | 2,924.0149 |
| `amihud_illiq` | 799,337 | 0 | 0.1356 | 0.3405 | 0.0000 | 0.0017 | 0.0128 | 0.1008 | 35.6286 |
| `volatility` | 793,738 | 5,599 | 0.1815 | 0.1570 | 0.0000 | 0.0913 | 0.1501 | 0.2356 | 39.5428 |
| `price_info` | 779,523 | 19,814 | 1.4296 | 1.6981 | –6.9068 | 0.3847 | 1.4762 | 2.5714 | 6.9068 |
| `ind_own` | 241,199 | 558,138 | 0.0686 | 0.0544 | 0.0098 | 0.0265 | 0.0510 | 0.0952 | 0.9526 |
| `r_ins_own` | 498,480 | 300,857 | 0.4690 | 0.3135 | 0.0000 | 0.1872 | 0.4760 | 0.7286 | 10.0599 |

## Notes
- The tabulated counts reveal that `ind_own` and `r_ins_own` are available for roughly 30% and 62% of the panel, respectively, while `volatility` and `price_info` are nearly complete.
- `price_info` is centered around 1.48 with a symmetric spread and is defined whenever price- or return-based signals are present; negative values arise from the underlying transformation.
