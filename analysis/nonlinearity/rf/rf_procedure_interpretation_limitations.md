# Random Forest Nonlinearity: Procedure, Interpretation, and Limitations

## Procedure

The random forest analysis is used as a flexible nonparametric check on the nonlinear relation between index ownership (`ind_own`) and market quality. The outcome variables are `amihud_illiq`, `volatility`, and `price_info`, where lower illiquidity and lower volatility are interpreted as improvements in market quality, while higher `price_info` is interpreted as an improvement. The estimation is run on the common complete-case sample so that the three outcome curves can be compared on the same observation set. To reduce the influence of extreme observations and keep the analysis stable, the script uses a dated train/evaluation split, a limited per-date training sample, and a modest bagged-tree ensemble.

For each outcome, the model includes `ind_own`, `c_firm_size`, `c_dollar_vol`, a date index, and firm/date mean context variables. The fitted forest is then used to generate partial dependence curves over a grid of `ind_own` values. The outcome-specific curves are converted into changes relative to the median ownership level, standardized, and combined into a composite score:

`score(x) = -z(illiquidity curve) - z(volatility curve) + z(price_info curve)`

The optimal indexing level is the value of `ind_own` that maximizes this composite score, and the high-quality range is defined as the top 5 percent of the score distribution across the grid.

## Interpretation

The RF results are best read as a flexible robustness check for nonlinearity, not as a causal estimate. The main takeaway is whether the fitted score peaks at an interior ownership level rather than at the extremes of the support. In this project, the composite RF score indicates that moderate indexing is associated with better overall market quality than very low or very high indexing. The exact peak should not be interpreted as a sharp structural threshold, because tree ensembles often produce stepwise or plateau-shaped fitted curves rather than smooth analytic turning points.

The RF output is also useful channel by channel. If the partial dependence curve for an outcome slopes in the economically favorable direction and then flattens or reverses, that supports a diminishing-returns interpretation. If the curve is nearly flat, the evidence for that channel is weak. When the three outcome curves are combined, the composite optimum reflects a tradeoff across liquidity, volatility, and price informativeness rather than a single outcome alone.

## Main Figure

The main figure is [rf_indexing_score.svg](/Users/xiang/github/Index_Fund/analysis/nonlinearity/rf/rf_indexing_score.svg), which plots the composite market-quality score against `ind_own`. The key feature of this figure is that the score is highest in a low-to-moderate indexing region rather than at the extremes, which supports an interior-optimum interpretation. In other words, the figure suggests that some indexing improves overall market quality, but the marginal benefit does not increase monotonically as passive ownership rises.

The shape is also informative because it is not driven by a single sharp peak. Instead, the score is relatively flat near the top, which means the economically relevant message is the range of “good” indexing values rather than one exact point estimate. For this reason, the top 5 percent band should be read as the preferred conclusion from the figure. The appendix-style panel [rf_publication_diagnostics.svg](/Users/xiang/github/Index_Fund/analysis/nonlinearity/rf/rf_publication_diagnostics.svg) complements this main figure by showing that the importance ranking, fit quality, residuals, and ICE curves are all consistent with the same broad nonlinear pattern.

## Limitations

The main limitation is that this is a predictive machine-learning exercise, not a causal design. The forest does not solve endogeneity in index ownership, and it should not be treated as evidence of causal effects without an identification strategy. A second limitation is that standard random forests do not naturally absorb firm and time fixed effects, so the panel structure must be handled through sample design and context features rather than exact FE estimation.

Another limitation is statistical power. Market-quality variables are noisy at the firm-date level, and the available predictors are relatively limited, so predictive fit is only moderate for some outcomes. In particular, volatility and price informativeness are harder to forecast than illiquidity. The custom forest used here is also deliberately small and conservative so that it can run in the current environment, which makes it suitable for robustness analysis but not for state-of-the-art prediction benchmarking.

Finally, the composite score depends on standardization and sign conventions. That is appropriate for comparing nonlinear shapes across heterogeneous outcomes, but the resulting optimum is therefore a summary statistic rather than a directly estimated structural parameter. For that reason, the RF result should be presented as supporting evidence that indexing has a nonlinear interior optimum, alongside the quadratic and spline specifications, rather than as the single definitive estimate.
