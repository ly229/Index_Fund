# IV robustness grid (cutoff=1000/2000)

## Procedure
- Ran `analysis/check/iv_robustness_grid.py` with `PYTHONPATH=analysis/IV_result/IV_codes` so that `panel_iv_rank_cutoff` can be imported.  
- The script sweeps two cutoffs (1,000 and 2,000) through bandwidths [50, 100, 150, 200, 300, 500] plus the global sample; it logs `first_stage_f`, `beta_endog`, and related diagnostics for the three dependent variables before plotting.
- Output artifacts:  
  * `analysis/iv_robustness_grid_results.csv` (numerical table)  
  * `analysis/iv_robustness_first_stage_f.svg` (first-stage F across windows)  
  * `analysis/iv_robustness_beta.svg` (IV coefficient across windows)

## Key takeaways
- **First-stage strength**: With either cutoff, the IV is extremely weak once we restrict to narrow local windows—F-statistics for bandwiths 150–500 stay well below 5 and often under 1, whereas the global samples deliver F > 25 (cutoff=1,000) or F > 130 (cutoff=2,000). The smallest windows (50/100) also produce only F ≈ 0.01–2.5, so none of the local windows meet the conventional F > 10 rule, indicating weak identification in the local variants.
- **IV coefficient instability**: The 2SLS coefficients on `ind_own` swing wildly (and often have enormous standard errors) inside the local windows—sometimes even flipping sign—while the global specifications provide more stable, statistically significant estimates (visible as smooth, bounded lines in the SVG). This noise mirrors the weak first stage and cautions against relying on the local estimates for inference.
- **Runtime warnings**: The script emitted NumPy warnings about division/overflow in the matrix routines; they originate from the weak instruments and extreme parameter estimates in the local windows. None of those warnings prevented the grid from finishing, but they reinforce that the matrices are numerically unstable when the instrument is too weak.

## Next steps
1. Use the global cutoffs (especially 2,000) for IV inference; the local versions lack first-stage strength.  
2. If you still want a local robustness check, consider shrinking the bandwidth grid further (e.g., 20, 30) while logging the first-stage F to see whether an intermediate window can balance identification and locality.

## Local-window fit plot
- `analysis/IV_robutness_check/iv_local_windows_fitted.svg` overlays the fitted IV slopes for the five selected local bandwidth specs. The lines are ordered from the weakest instruments (cutoff=2000, bandwidth=50/100) through the median (cutoff=2000, bandwidth=300) to the strongest locals (cutoff=1000, bandwidth=50/100), with the legend and note positioned under the x-axis to keep the main panel area uncluttered.
- Without panel titles and with the y-axis labels now matching the dependent variables, the figure emphasizes how the weakest specs produce nearly flat or noisy slopes while the stronger ones return the steeper, more stable lines needed for credible inference. This visual reinforces why only the best-first-stage local windows resemble the global benchmark and are worth interpreting.
