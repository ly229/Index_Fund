# Nonlinearity Summary (Including IV)

## 1) Objective
Assess whether the effect of indexing (`ind_own`) on market outcomes is nonlinear, and answer:
**How much indexing is good for market quality?**

Outcomes:
- `amihud_illiq` (lower is better)
- `volatility` (lower is better)
- `price_info` (higher is better)

---

## 2) Quadratic Panel FE (Non-IV)
Model:
- `y_it = b1*ind_own_it + b2*ind_own_it^2 + controls + firm FE + time FE`
- Two-way clustered SE (firm, date)

Main estimated nonlinear patterns:
- `amihud_illiq`: `b2 > 0` and significant; marginal effect is generally positive in support.
- `volatility`: U-shape (`b1 < 0`, `b2 > 0`, both significant), turning point in support.
- `price_info`: U-shape (`b1 < 0`, `b2 > 0`, both significant), turning point near upper support.

Composite market-quality score construction:
- Standardize predicted quadratic changes for each outcome over an `ind_own` grid.
- Score = `-z(amihud) - z(volatility) + z(price_info)`.

Core non-IV answer:
- Composite optimum: `ind_own ≈ 0.0379`
- Top-score range (top 5%): `[0.0328, 0.0430]`

Interpretation:
- Non-IV evidence favors a **low-to-moderate indexing level** as optimal in this sample.

---

## 3) Quadratic IV Design
2SLS nonlinear specification:
- Endogenous: `ind_own`, `ind_own^2`
- Excluded instruments: `z_cutoff * running^k`, `k=0..2`
- Controls: `c_firm_size`, `c_dollar_vol`, `running`, `running^2`
- Firm FE + time FE via two-way demeaning

### A) Global sample, cutoff = 2000 (preferred IV)
File: `iv_quadratic_cutoff2000_global.csv`

First-stage strength (joint F):
- `ind_own`: very strong (~600+)
- `ind_own^2`: strong (`~25-34`)

Second-stage highlights:
- `volatility`: strong U-shape
  - `b1 = -2.5336` (p=0.000186)
  - `b2 = 15.6492` (p=0.000380)
  - turning point `~0.0809` (in support)
- `amihud_illiq`:
  - `b2 = -4.8236` (p=0.0366), `b1` not significant
  - turning point `~0.0343` (in support)
- `price_info`: not statistically significant nonlinear effect

Conclusion for IV(2000):
- Credible IV nonlinearity is strongest for `volatility`.
- Evidence for `amihud_illiq` nonlinearity is weaker/mixed.
- No robust IV nonlinearity for `price_info`.

### B) Global sample, cutoff = 1000 (not preferred)
File: `iv_quadratic_cutoff1000_global.csv`

Issue:
- First-stage diagnostics became numerically unstable/invalid in parts of the workflow (including negative/ill-behaved F values in comparison outputs).
- This indicates weak/unstable identification and reduces economic interpretability.

Economic interpretation:
- `cutoff=1000` is **not economically reliable** in this setup.
- Keep it as a failed/weak robustness check, not as a main causal result.

---

## 4) 1000 vs 2000 IV Comparison
File: `iv_quadratic_cutoff1000_vs_2000_comparison.csv`

Summary:
- `cutoff=2000` delivers stable and interpretable first-stage/second-stage behavior.
- `cutoff=1000` is unstable and not suitable as baseline nonlinear IV evidence.

---

## 5) Visual Outputs
Non-IV:
- `indexing_good_range_score.svg`
- `nonlinear_relationship_all3.svg`

IV (cutoff=2000):
- `iv_quadratic_coefficients_cutoff2000_global.svg`
- `iv_quadratic_curves_cutoff2000_global.svg`
- `iv_quadratic_fitted_lines_cutoff2000_global.svg`
- `iv_first_stage_cutoff2000_global.svg`
- `iv_fitted_indownhat_vs_depvars_cutoff2000_global.svg`

IV (cutoff=1000, robustness style plots):
- `iv_first_stage_cutoff1000_global.svg`
- `iv_fitted_indownhat_vs_depvars_cutoff1000_global.svg`

---

## 6) Final Takeaway
- Nonlinearity is a first-order feature in this project.
- Reduced-form FE suggests an interior optimum around `ind_own ~ 0.04`.
- In IV space, **cutoff=2000 global** is the only specification here with strong enough first stage to support nonlinear inference.
- Preferred economic statement:
  - Indexing appears beneficial for volatility at low-to-moderate levels, with diminishing/reversal behavior at higher levels (U-shape), while other channels are weaker or less stable in IV.
