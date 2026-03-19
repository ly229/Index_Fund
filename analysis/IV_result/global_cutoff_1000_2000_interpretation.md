# Interpretation of Global IV Plots: Cutoff 1000 vs Cutoff 2000

This note interprets the two global IV fitted-line plots in [`analysis/IV_result`](./) and compares the estimation results reported in [`iv_results_global_cutoff_1000_2000.csv`](./iv_results_global_cutoff_1000_2000.csv).

## Global cutoff = 1000

The global `cutoff = 1000` specification delivers a relevant but only moderately strong first stage, with first-stage F-statistics between approximately 27 and 28. In the second stage, the estimated coefficient on `ind_own` is positive for `amihud_illiq` (`beta = 0.232`, `p = 0.110`), negative for `volatility` (`beta = -1.485`, `p < 0.001`), and negative for `price_info` (`beta = -5.881`, `p = 0.143`). Since a lower Amihud illiquidity measure implies greater liquidity, the positive but insignificant coefficient suggests no reliable liquidity effect in this specification. By contrast, the negative and statistically significant volatility estimate implies that higher instrumented ownership is associated with lower volatility. The negative `price_info` estimate is not statistically distinguishable from zero, so the evidence for an effect on price informativeness remains weak.

In terms of the plot, the red fitted line is most informative for the volatility panel, where the slope is clearly downward. The fitted lines for illiquidity and price informativeness are less persuasive because the coefficients are imprecisely estimated. Therefore, the cutoff-1000 specification points primarily to a volatility-reduction channel rather than a robust liquidity or informativeness effect.

## Global cutoff = 2000

The global `cutoff = 2000` specification is substantially stronger from an identification perspective. The first-stage F-statistics are around 129 to 131, which is far stronger than in the `cutoff = 1000` case and indicates that the instrument has much greater predictive power for `ind_own`. In the second stage, the coefficient on `ind_own` is negative and statistically significant for `amihud_illiq` (`beta = -0.285`, `p < 0.001`), negative but insignificant for `volatility` (`beta = -0.177`, `p = 0.162`), and strongly negative and highly significant for `price_info` (`beta = -12.087`, `p < 0.001`). The negative Amihud coefficient implies improved liquidity, while the strongly negative `price_info` coefficient suggests a decline in the measured price informativeness proxy.

In the plot, the downward-sloping fitted lines for `amihud_illiq` and `price_info` are the central visual result. The volatility panel still slopes downward, but the estimate is too imprecise to support a strong conclusion. Accordingly, the cutoff-2000 specification suggests that higher instrumented ownership improves liquidity but may reduce measured price informativeness, while the volatility effect is not robust.

## Comparison of cutoff 1000 and cutoff 2000

The comparison between the two global cutoff specifications is economically important because the substantive conclusions differ materially across the two thresholds. At the Russell 1000 cutoff, the dominant result is a negative and statistically significant volatility effect, whereas the estimated effects on illiquidity and price informativeness are not statistically significant. At the Russell 2000 cutoff, the volatility effect weakens and becomes insignificant, but the liquidity and price-informativeness effects become much stronger in both magnitude and statistical precision.

This shift matters because the `cutoff = 2000` specification also has a much stronger first stage. As a result, the `cutoff = 2000` estimates are more credible as IV evidence. Taken together, the two plots suggest that the main robust effect of instrumented ownership is not a reduction in volatility, but rather a combination of improved liquidity and lower measured price informativeness. Put differently, the global `cutoff = 1000` results point to one potential channel, but the global `cutoff = 2000` results provide the more persuasive evidence and imply that the primary effect operates through trading conditions and the information environment.

## Short write-up version

Using the global cutoff-IV design, the results differ sharply across the two index thresholds. At the Russell 1000 cutoff, instrumented ownership is associated primarily with lower volatility, while the estimated effects on illiquidity and price informativeness are statistically weak. In contrast, at the Russell 2000 cutoff, the first stage becomes much stronger and the second-stage estimates indicate significantly lower Amihud illiquidity and significantly lower price informativeness, whereas the volatility effect becomes small and insignificant. Therefore, the Russell 2000 specification provides the more credible evidence and suggests that the dominant effect of index-related ownership operates through improved liquidity rather than reduced return volatility.
