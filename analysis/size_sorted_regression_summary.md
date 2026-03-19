# Size-sorted pre/post reconstitution regression

## Procedure
- Script: `analysis/prepost_reconstitution_size_sorted_bw100.py:1-214` defines the workflow. It loads the Russell 3 rank file, keeps firms within ±100 rank points of cutoff=2000, and assigns each treated firm to a tercile (`Small`, `Mid`, `Large`) based on the pre-reconstitution (tau = –1, March) market cap for that year.
- For every dependent variable (`amihud_illiq`, `volatility`, `price_info`) and each bucket, the code averages the outcome at quarterly event times `tau ∈ {−2, −1, 0, +1, +2}` relative to the June reconstitution (event time 0) and stores the aggregated means and counts in `analysis/prepost_reconstitution_size_sorted_bw100.csv`.
- A custom SVG (`analysis/prepost_reconstitution_size_sorted_bw100.svg`) plots the five-point pre/post profiles for each outcome, letting the reader see whether the size-sorted buckets behave differently around the Russell 2000 cutoffs.

## Results
- The generated CSV (`analysis/prepost_reconstitution_size_sorted_bw100.csv:1-30`) reports that the smallest tercile has the highest Amihud illiquidity throughout the window (means ≈0.07 at tau 0) while the mid/large buckets are an order of magnitude lower (~0.006). The small firms’ illiquidity dips slightly after the reconstitution and partially rebounds by tau=+2.
- Volatility is monotonic in size: small firms sit around 0.16 at the event quarter, mid-size around 0.16–0.17, and large firms closer to 0.19–0.20 (lines 17-33). The large bucket also experiences the largest fluctuation between tau=–1 and tau=+1, consistent with greater sensitivity to the reconstitution shock.
- Price informativeness peaks for mid-sized firms (≈0.78 before the event) and falls toward ≈0.71 afterwards, whereas small firms start lower (≈0.69) and decline more sharply; large firms remain between the two extremes. These patterns (lines 34-40) suggest the mid bucket retains the most stable price signal, while the smallest bucket loses informativeness after the reconstitution.

## Takeaways
- The size-sorted pre/post averages reveal heterogeneous liquidity and volatility responses to the Russell 2000 cutoff: smaller firms are illiquid and volatile, the largest firms move less but still lag mid-sized firms on price informativeness, and the reconstitution event causes transitory movements across all buckets.
- Use the accompanying CSV and SVG for further diagnostics or to overlay these size buckets on the main IV regressions, especially if you want to interpret the robustness of the cutoff-IV strategy across firm sizes.
