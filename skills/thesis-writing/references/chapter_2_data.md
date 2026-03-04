<!-- Page 8 -->

2.3 Volatility
Some studies have tested how ETF ownership causes a rise of volatility for the
underlying stocks. Ben-David et al.(2018) find that stocks with higher ETF
ownership display higher volatility. Sullivan and Xiong (2012), Gregoire (2019),
and Chabakauri and Rytchkov(2021) show that higher indexing can make stocks
more volatile and more correlated. Anadu et al(2020), Rossi and Steliaros(2022)
document that higher index ownership amplifies market volatility. My finding
can provide more empirical support for this trend of literature.
3 Data
Wharton Research Data Services(WRDS) is a web-based business data research
service from the Wharton School, University of Pennsylvania. To construct
index and institutional ownership, I use the holding data from Thomson Reuters
Mutual Fund Holdings(S12). Thomson database covers almost all historical
domestic mutual funds since 1980. Hence it is a comprehensive database and
largely free of survival bias. For market quality measures and control variables,
I use the daily stock file from Center for Research in Security Prices(CRSP).
It provides detailed trading data for the daily stock market, for instance, bid
and ask prices, open and close prices, high and low prices, trading volume,
etc. Both Thomson and CRSP own data for a long horizon back to 2000.
Hence, I can use them for future panel regressions. One issue with these two
datasets is that they are in different time frequencies. Thomson provides holding
information at the end of each quarter while the stock information gathered
by CRSP is at daily frequency. To minimize the potential impact induced by
autocorrelation problems in the time series, I introduced a few lag variables into
control variables. My current focus is on a cross-sectional regression within the
time period of the first quarter in 2022. This will be extended to a longer time
horizon including the period before COVID recession. Then I can test for two
impacts. The first is to find how changes in index ownership over time cause
changes in the market quality. Secondly, I can examine how the impact of index
ownership on market quality changes under bad financial circumstances.
7

<!-- Page 9 -->

3.1 Filter for Indexing
To separate index funds and ETFs from other mutual funds, I apply a filter on
the fund names - grasping funds whose names contain “IND” or “ETF”. To see
the quality of this filter, I run a few sample tests. I define the event “Positive”
as the fund is index funds or EFTs. Therefore, false positive is the outcome
that the filter incorrectly categorizes other funds as index funds; false negative
is the outcome when the filter fails to capture the real index funds or ETFs.
By taking a few random samples from my dataset and checking whether they
are real index funds or ETFs, I find the false positive rate is 5% for “IND”
and 1% for “ETF”. By checking the largest 100 index funds and ETFs to see
whether my filter successfully captures them, I find that the false negative rate
is 1% for “IND” and 3% for “ETF”. Thus, I think my filter has an overall good
performance on separating index funds and ETFs from other mutual funds.
4 Preliminary Analysis
This section is to examine the relationship between index ownership and market
quality. Again, the market quality is referred to price informativeness, liquidity
and volatility of the stock. I start with cross-sectional regression of market qual-
ity on index ownership within the first quarter of 2022 (01/01/2022-03/31/2022).
The regression result shows that stock return synchronicity is positively corre-
lated with index ownership. All liquidity measures indicate that index owner-
ship leads to a more liquid market except for the result from shares turnover.
Volatility is amplified by a higher index ownership.
4.1 Baseline Regression Model
I run the following regression to measure the relationship between market qual-
ity(price informativeness, liquidity and volatility) and index ownership:
Market Quality i,q=α+βIndex i,q+γXi,q+ϵi,q (1)
where Market Quality i,qincludes stock return synchronicity( SY NCH i,q),
8