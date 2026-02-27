# Baseline Panel OLS Interpretation

Specification: firm and time fixed effects with two-way clustered SE (firm, date).

## Main variable: `ind_own`
- Amihud illiquidity: positive association, coef=0.0613, SE=0.0201, t=3.05, p=0.002327, 95% CI=[0.0219, 0.1008] (1%).
- Volatility: negative association, coef=-0.0560, SE=0.0233, t=-2.40, p=0.01634, 95% CI=[-0.1018, -0.0103] (5%).
- Price informativeness: negative association, coef=-3.3018, SE=0.2384, t=-13.85, p=0, 95% CI=[-3.7691, -2.8345] (1%).

## Controls
- Firm size (c_firm_size)
  - Amihud illiquidity: positive, coef=0.0084, p=3.67e-07.
  - Volatility: negative, coef=-0.0984, p=0.
  - Price informativeness: negative, coef=-0.3406, p=0.
- Dollar volume (c_dollar_vol)
  - Amihud illiquidity: negative, coef=-0.0222, p=0.
  - Volatility: positive, coef=0.0748, p=0.
  - Price informativeness: positive, coef=0.1453, p=0.

## Sample size
- Amihud illiquidity: N=241,199, firms=8,877, periods=100.
- Volatility: N=240,515, firms=8,858, periods=100.
- Price informativeness: N=239,758, firms=8,855, periods=100.
