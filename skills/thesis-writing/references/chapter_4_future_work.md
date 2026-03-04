<!-- Page 13 -->

6 Future Work
6.1 Causal Effect
One limitation of the regression result in Table 2 is that it only reflects the
correlation between index ownership and market quality. I need to address the
endogeneity problem in order to establish the causal effect of index ownership on
market quality. The endogeneity problem mainly comes from reverse causality.
The impact between index ownership and market quality can run the other way.
For example, a greater stock price informativeness can lead more investors to
hold index funds. To confirm that index ownership is the cause of changes in
market quality, I will use the exogenous index ownership changes caused by
Russell 1000/2000 reconstitutions or S&P500 index additions as instruments.
The Russell 1000 and Russell 2000 indices list all companies in descending
order by their market capitalization: the Russell 1000 tracks the largest 1,000
stocks and the Russell 2000 includes the next 2,000 smaller stocks traded in the
United States. Reconstitution happens once a year following a mechanical rule.
Some stocks switched from the Russell 2000 to the Russell 1000 experienced an
increase in their market capitalization while others switched from the Russell
1000 to the Russell 2000 experienced a shrink in their market capitalization. The
12

<!-- Page 14 -->

index ownership for the top stocks in the Russell 2000 is higher than the bottom
ones in the Russell 1000 after reconstitution, despite the fact that stocks in the
latter group have larger market capitalization than those in the former group.
Researchers like Ben-David et al.(2018) apply an identification strategy based
on the idea that the Russell 1000/2000 reconstitutions cause exogenous changes
in the index ownership. This change in index ownership is purely caused by the
Russell 1000/2000 reconstitutions and is not related to other market factors like
market capitalization.
The logic of using the S&P500 index addition as instrument is similar to
the one stated above. However, not many studies have been done before using
the S&P500 index addition to identify the causal effect of index ownership on
market quality. Sammon(2022) is the one I know who uses the S&P500 index
addition as instrument to find the causal impact of index ownership on price
informativeness. So I need to do careful examination before using it for my
identification strategy. As the most well-know index in the U.S., it is clear that
adding to S&P500 index leads to a rise of the index holdings for the underlying
stocks. So it satisfies the relevance condition. To prove the exclusion condi-
tion, I will select a sample of firms who are very likely to join S&P500 index
and compare their trading characteristics. This can be done by examining the
past price informativeness, liquidity and volatility of newly joined companies
to S&P500. If the market quality measures of these stocks exhibit nice hetero-
geneity, then the exclusion condition is proved, otherwise, this instrument is not
strong enough for identification.
6.2 Panel Regression
My baseline regression is a cross-sectional regression. In order to find the effect
of a change in index ownership on market quality, I need to extend it to a panel
regression with a longer time horizon. My plan is to conduct the panel regression
from 2018 to 2022. This includes several important financially stressful events,
the COVID recession and 2020 stock market crashes. As Saglam et al(2020)
document that the impact of ETF ownership can revert under a stressful finan-
cial environment, I also want to explore how the impact of index ownership on
market quality would react to under different financial circumstances. And my
13