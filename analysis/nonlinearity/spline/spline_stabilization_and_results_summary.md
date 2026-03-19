# Spline Stabilization and Results Summary

## Stabilization Procedure

The spline stabilization in [spline_panel.py](/Users/xiang/github/Index_Fund/analysis/nonlinearity/spline_panel.py) had three parts.

First, `ind_own` was clipped to the empirical support used for interpretation, namely the pooled `[q01, q99]` range, and then rescaled to `[0,1]`. Economically, this keeps the fitted nonlinear effect focused on the part of the ownership distribution where the data are actually dense enough to identify the relationship. Numerically, it prevents the spline basis from being driven by extreme tail observations and reduces conditioning problems once firm and time fixed effects are absorbed.

Second, the interior knots were moved from tail-heavy locations to the center of the distribution, at the `25%`, `50%`, and `75%` quantiles. The economic logic is that the question is about the practically relevant range of indexing, not rare extreme ownership realizations. Statistically, central knots allocate flexibility where the sample has information, while avoiding unstable curvature in sparse tails.

Third, prediction was restricted to the same bounded support rather than allowing effective extrapolation. This matters because natural cubic splines can still produce misleading tail behavior if the figure extends beyond the well-identified region. The benefit is that the plots and “optimal indexing” range reflect interpolated patterns in the observed data, not functional-form artifacts at the extremes.

## Economic Reasoning

The broader economic reason for stabilizing the spline is that the research question is not “what happens at the most extreme observed ownership values,” but “how much indexing is good for market quality in the economically relevant range.” A flexible nonlinear model is useful only if its shape is learned from real variation in the sample. By concentrating the spline on the interior support, the estimated tradeoff between liquidity, volatility, and price informativeness becomes more credible and easier to defend in a thesis or paper.

This also improves interpretation. A stabilized spline is less likely to generate spurious turning points caused by a few high-ownership observations. That means the implied optimum is better understood as a genuine interior balance point between benefits and diminishing returns, rather than a mechanical byproduct of unstable tail curvature.

## Spline Results

The main numerical summary is in [spline_indexing_summary.md](/Users/xiang/github/Index_Fund/analysis/nonlinearity/spline_indexing_summary.md), with detailed coefficients in [spline_panel_results.csv](/Users/xiang/github/Index_Fund/analysis/nonlinearity/spline_panel_results.csv) and the composite score grid in [spline_indexing_score_grid.csv](/Users/xiang/github/Index_Fund/analysis/nonlinearity/spline_indexing_score_grid.csv).

The stabilized spline gives a composite optimum at about `ind_own ≈ 0.0548`, with a top-quality range of roughly `[0.0500, 0.0603]`. Economically, this suggests that market quality appears best at low-to-moderate indexing levels, around 5% to 6% in this sample. Relative to the earlier quadratic result, the spline pushes the optimum slightly upward, but the core conclusion is unchanged: more indexing is not monotonically better, and the best region is interior rather than at the extremes.

The outcome-specific results show why that composite optimum emerges. For `amihud_illiq`, the best point is around `0.0467`, which suggests liquidity improves as indexing rises into the moderate range. For `volatility`, the best point is around `0.0733`, indicating that volatility reduction continues somewhat further into the ownership distribution. For `price_info`, the best point is near the lower bound of the support, around `0.0104`, which implies that very high indexing is not associated with stronger price informativeness in this descriptive spline specification. The composite optimum near `0.055` is therefore a compromise across these channels.

## Plot Interpretation

The score plot [spline_good_range_score.svg](/Users/xiang/github/Index_Fund/analysis/nonlinearity/spline_good_range_score.svg) is the clearest visual answer to “how much indexing is good for the market?” The peak of the curve marks the ownership level where the standardized combination of lower illiquidity, lower volatility, and higher price informativeness is strongest. The shaded band gives the top 5% range, which is useful because it avoids over-interpreting a single point estimate and instead emphasizes a plausible interval of “good” indexing levels.

The multi-panel spline plot [spline_relationship_all3.svg](/Users/xiang/github/Index_Fund/analysis/nonlinearity/spline_relationship_all3.svg) shows the channel-by-channel patterns. The fitted line in each panel is the predicted change relative to the median ownership level, and the 95% confidence band shows where the shape is estimated more or less precisely. The main reading is:

- Illiquidity improves into the moderate-indexing region.
- Volatility also improves at low-to-moderate indexing, with the strongest gains extending a bit farther right.
- Price informativeness does not show the same interior optimum and appears weaker or less supportive of higher indexing.

Taken together, the spline evidence supports an interior-optimum story: some indexing is beneficial for market functioning, but beyond a moderate range the incremental benefits flatten or become less favorable across dimensions. The main caution is that this remains descriptive non-IV evidence, so it is best framed as a robust nonlinear pattern rather than a fully causal threshold.
