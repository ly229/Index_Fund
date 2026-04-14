# Random Forest Nonlinearity: How Much Indexing Is Good?

Specification:
- Bagged regression trees estimated in pure Python/Numpy because `scikit-learn` is unavailable in this environment.
- Main predictor: `ind_own`.
- Controls: `c_firm_size`, `c_dollar_vol`, date index, and firm/date mean context variables.
- The model is a flexible nonparametric robustness check, not a causal design.

## Sample and Fit
- `amihud_illiq`: train n=12000, eval n=43311, pd n=4800, train R2=0.396, eval R2=0.362
- `volatility`: train n=12000, eval n=43311, pd n=4800, train R2=0.172, eval R2=0.169
- `price_info`: train n=12000, eval n=43311, pd n=4800, train R2=0.224, eval R2=0.183

## Composite Answer
- Composite optimum indexing level: `ind_own ≈ 0.0270`
- High-quality range (top 5% score): `[0.0270, 0.0438]`

## Reading the Curves
- Lower `amihud_illiq` and `volatility` are better.
- Higher `price_info` is better.
- The score combines the three standardized partial-dependence curves with the appropriate signs.

## Diagnostic Plots
- `rf_feature_importance.svg`: permutation importance by outcome.
- `rf_observed_vs_predicted.svg`: observed vs fitted values for train and eval samples.
- `rf_residual_diagnostics.svg`: residual histograms and residual-vs-fitted plots.
- `rf_ice_curves_ind_own.svg`: individual conditional expectation curves for `ind_own`.
- `rf_publication_diagnostics.svg`: compact appendix-style panel for the RF model.
