# Quadratic IV - Global Sample, Cutoff 2000

- Endogenous: ind_own, ind_own^2
- Excluded IV: z_cutoff * running^k, k=0..2
- Controls: c_firm_size, c_dollar_vol, running, running^2
- FE: firm + date (two-way demeaning)
- Sample: global (bandwidth effectively unrestricted)

## amihud_illiq
- b1=0.3313 (p=0.385), b2=-4.824 (p=0.0366)
- turning=0.0343, in_support=True [q01=0.0104, q99=0.2269]
- first-stage joint F: ind_own=638.786 (p=0); ind_own^2=28.413 (p=0)

## volatility
- b1=-2.534 (p=0.000186), b2=15.65 (p=0.00038)
- turning=0.0809, in_support=True [q01=0.0104, q99=0.2269]
- first-stage joint F: ind_own=611.060 (p=0); ind_own^2=25.592 (p=1.11e-16)

## price_info
- b1=-6.801 (p=0.285), b2=-35.91 (p=0.401)
- turning=-0.0947, in_support=False [q01=0.0104, q99=0.2268]
- first-stage joint F: ind_own=664.395 (p=0); ind_own^2=34.308 (p=0)
