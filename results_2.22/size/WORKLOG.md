# Work Summary

## Overview
- Ran size-sorted (mktcap decile) panel regressions with two-way fixed effects and two-way clustered SEs, focusing on `ind_own` effects.
- Generated scatter plots and decile-beta plots (with confidence bands) for `amihud_illiq`, `volatility`, and `price_info`.
- Produced size-sorted regression summaries (TXT + Markdown with p-values).
- Committed and pushed results to `origin/main`.

## Key Outputs
- Plots:
  - `amihud_illiq_vs_ind_own.svg`
  - `amihud_illiq_vs_ind_own_top10.svg`
  - `price_info_vs_ind_own_top10.svg`
  - `results_2.22/size/beta_vs_decile_amihud_ci.svg`
  - `results_2.22/size/beta_vs_decile_volatility_ci.svg`
  - `results_2.22/size/beta_vs_decile_price_info_ci.svg`
- Tables:
  - `results_2.22/size/size_sorted_regression_summary.txt`
  - `results_2.22/size/size_sorted_regression_summary.md`

## Methods
- Two-way fixed effects (cusip + date), controls: `c_firm_size`, `c_dollar_vol`.
- Two-way clustered SEs (cusip × date).
- Decile sorting done on observation-level `mktcap`.

## Git
- Commit: `4790235` — "Add size-sorted regression plots and summaries"
- Pushed to: `origin/main`
