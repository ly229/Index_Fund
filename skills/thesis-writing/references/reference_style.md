

<!-- Page 1 -->

How much indexing is good for the market?
by
Lu Yu
A dissertation submitted in partial satisfaction of the requirements for the
degree of
Doctor of Philosophy
in the Graduate Division of Economics
Georgetown University
Supervisors: Professor James Angel
Professor Toshihiko Mukoyama
Panel Member: Professor Ivana Komunjer
Professor Behzad Diba
Professor Dan Cao
November, 2022
©Lu Yu, 2022


<!-- Page 2 -->

1 Introduction
The growth of index investing is remarkable in the U.S. stock market. Introduc-
tion of the first index mutual funds was in the 1970s, followed by index ETFs
in the 1990s. By the end of 2020, total net assets of these two index funds
had grown to $9.96 trillion, which is the total net asset value of the 2010 fund
market. As Figure 1 shows, in 1993, index equity mutual funds was a trivial
player in the equity mutual fund market, whereas it took over approximately
one quarter of the market in 2020. Because of the popularity of indexing, active
investing is losing its shares in the market. From 2011 through 2020, index
domestic equity mutual funds and ETFs received $1.9 trillion in net new cash
and reinvested dividends, while actively managed domestic equity mutual funds
experienced net outflows of almost the same amount.1
Figure 1: The Rise of Index Investing. It shows the changes of equity mutual fund
market composition from 1993 to 2020. Blue line shows a decreasing trend of actively managed
equity mutual funds while the orange line depicts an increasing popularity of index investing
in the equity mutual fund market from 1993 to 2020. Data Source: ICC Fact Book (Year
2005-2021), Table42
The notable rise of index investing warrants a careful examination of its im-
1The data are for the U.S. market and from Page 48-49 of the 2021 Investment Company
Institute(ICI) Fact Book. ICI is the trading association representing regulated investment
funds. It is the primary source for statistical data on the investment company industry,
including mutual funds, exchange-traded funds, closed-end funds, and UITs.
1


<!-- Page 3 -->

pact on the financial markets. Does higher index ownership benefit or hurt the
trading market quality? Of great importance to investors in decision making is
how index funds influence the stock market environment. For example, whether
a higher index ownership makes the stock market more transparent; whether
increased indexing leads to a more liquid market; whether more index investing
increases volatility of the underlying stocks.
Price informativeness refers to the precision of the private signals on the
fundamentals of asset returns revealed by prices. Higher price informativeness
means a more transparent trading market. Traditionally, people think as index
funds become more popular in the trading market, active investors intend to
switch to passive investors. It means less information search and lower market
transparency. Nevertheless, this argument ignores the inflow of new traders
attracted by index funds as well as the influence of lower transaction costs of
index funds on existing investors.
Along with the growing popularity of index investing, there is a well-documented
trend of increasing price informativeness (Bai et al. (2016); Brogaard et al.(2019);
Davila and Parlatore (2020); MartineauMartineau2021). Proponents of index
investing vehicles argue that there is a positive link between these two increasing
trends. As index funds introduce investors to a range of diversified portfolios at
lower information costs, more investors, both active and passive, are attracted
to the market. Meanwhile, reduced searching costs lead to a more efficient
information acquisition and incorporation process. Kim et al.(2012), Dong et
al.(2016) find that reduced information processing costs can increase price in-
formativeness of the affected stocks. As a result, higher index ownership can
lead to higher price informativeness. My baseline regression result supports
the argument that there is a positive link between index ownership and price
informativeness.
If the rise of index funds causes a lower concentration of active investors as
they shift to passive investors, the market liquidity will be impaired. However,
as index funds provide a wider range of various portfolios with lower costs, more
investors are brought into the stock market. The inflow can be both passive and
active investors. Becoming new uninformed traders is a good choice for investors
who are hesitant to engage in the stock market if their intention is to place a
2


<!-- Page 4 -->

diversified “bet” with low transaction cost and small energy input. The arrival
of new investors increases liquidity of the underlying stocks Holden(1995). At
the same time, the lower transaction and information costs brought by index
funds may lead active investors to trade more aggressively and contribute to a
more liquid market. My results of all measurements on liquidity(bid-ask spread,
trading volume, and Amihud illiquidity measurement) show that higher index
ownership leads to higher liquidity of the underlying stock, except for the shares
turnover measure. This remains puzzling for now but will be interesting to figure
out what causes the opposite result.
Index funds bring more passive investors who have a “demand elasticity” close
to zero. They do not respond to the price signal as timely as active traders do.
Haddad et al.(2022) found that as the passive investors take larger shares of
the stock market, the aggregate demand curve has become substantially more
inelastic for individual stocks. When demand is price inelastic, a small increase
in supply induces a large fall in price and makes price more volatile. The coeffi-
cient from my baseline regressions indicates a positive correlation between index
ownership and idiosyncratic volatility. The magnitude of coefficient shows that
it is economically significant however the p-value tells a statistically insignificant
story.
1.1 Research Question
The research question is to explore the impact of index ownership on the stock
market. Does the increase of index investing improve or impair the market
quality? Which market provides a better environment for the investor to make
decisions, a market with no indexing or a market full of indexing? I choose three
main proxies to reflect the quality of a trading market. The first one is stock
price informativeness. It tells us how transparent the market is. Since price in-
formativeness is not always an easy measurement, I use stock return synchronic-
ity to measure the price informativeness of the underlying stock. Secondly, I
use bid-ask spread, trading volume, shares turnover, and Amihud illiquidity
measure to examine how index ownership affects liquidity. Lastly, I calculate
idiosyncratic return volatility from daily close prices.
3


<!-- Page 5 -->

The baseline regression result reports a positive correlation between index
ownership and stock synchronicity. Higher index ownership is associated with
tighter bid-ask spreads, higher trading volumes and smaller Amihud illiquidity
measure hence a more liquid trading environment for the underlying stocks.
However, shares turnover becomes smaller for stocks with larger index owner-
ship. The negative correlation between shares turnover and index ownership
indicates an opposite effect of index ownership on liquidity. The reason why
the regression coefficients on trading volume and shares turnover have opposite
signs remains puzzling for now and requires deeper examinations. A stock with
more index ownership exhibits higher volatility.
However, all the above result is a correlation measure. To quantify the causal
relationship between index ownership and market quality is empirically chal-
lenging as they are jointly determined in equilibrium. For example, an alter-
native interpretation for the relationship between index ownership and price
informativeness is that the causality runs the other way. An index fund is more
likely to include firms with larger market capitalizations. Larger firms are more
familiar to investors and exposed to more disclosures, so perhaps more infor-
mation is available to the public and incorporated into their prices. Hence they
have higher comovements with the return of S&P500 and higher stock return
synchronicity.
To establish the causal link between index ownership and market quality, I
will construct instruments for my baseline regression using exogenous changes
in index ownership due to the Russell 1000/2000 rebalancing (Ben-David et
al.(2018), Glossner(2020, Cole et al.(2020) and the S&P500 index additions
(Bennett et al.(2022), Sammon(2022). The underlying assumption is that index
rebalancing or addition only influences market quality through the channel of
index ownership.
4


<!-- Page 6 -->

2 Literature Review
2.1 Price Informativeness and Stock Synchronicity
There is still a considerable disagreement among researchers on the impact of
higher index investing on price informativeness. Glosten et al.(2019) find that
rise in the ETFs investment can improve the accuracy of accounting informa-
tion incorporated into stocks, implying increased price efficiency. Malikov(2019)
shows that higher shares of index investing can improve price efficiency when
stock pricking effect dominates the market timing effect. Buss and Sundare-
san(2020) and Lee(2020) support that the rise of index investing positively af-
fects price efficiency. On the other hand, Cong and Xu(2018), Bond and Gra-
cia(2019), Sammon(2022) find a negative correlation between the rise of index
investing and price informativeness.
Another stream of studies argues that changes in the market composition have
no or ambiguous effects on price efficiency. Grossman and Stiglitz (1980) and
Cole et al.(2020) prove that the fraction of informed and uninformed investors in
the market does not affect price efficiency as there are some cancel-out and auto-
adjusting effects. Liu and Wang (2018) show that the rising of indexing may
lead to increased or decreased price informativeness depending on the causes
of increasing index investing. If the rise of indexing is exogenous, then the
increasing index investing would lead to less informative prices.
Part of the reason for the debates among literature is that there is no unified
way of measuring price informativeness. As price informativeness is the notion
of how well the price of a stock reflects the fundamental value of the firm. The
traditional measure of price informativeness is to estimate the fraction of fun-
damentals revealed by stock prices, for instance, Grossman and Stiglitz(1980),
Cong and Xu(2018), Lee(2020). But a correct measure of fundamentals of the
firm remains debatable. Therefore, I try to look at price informativeness from
another aspect, which is to use stocks synchronicity to reflect the informative-
ness of stock price.
Stock return synchronicity measures the degree of which individual stock re-
turns comove with the return from market index. Dasgupta et al. (2010) prove
5


<!-- Page 7 -->

both theoretically and empirically that synchronicity increases in a more trans-
parent environment. Hou et al. (2013) proposes a theoretical model to show
that greater synchronicity indicates less idiosyncratic noises and higher informa-
tiveness in a noisy environment. Chan and Chan(2014) find evidence from the
pricing of seasoned equity offerings showing that the price is more informative
when stock return synchronicity is higher. Kan and Gong(2018) and Abedifar
et al.(2020) find that higher stock synchronicity implies higher price informa-
tiveness using DID analysis and a regression discontinuity design respectively.
As of my best knowledge, no previous researchers have utilized stock return syn-
chronicity as a measurement to explore the impact of index ownership on price
informativeness. My contribution to the literature is to provide more empirical
evidence supporting the argument that larger index ownership leads to higher
stock return synchronicity and better price informativeness.
2.2 Liquidity
Researchers have found competing evidence on how a larger index ownership
influences the liquidity of the underlying stocks. Hamm(2014) argues that
ETFs lead to a higher Kyle’s lambda hence lower liquidity. Israeli et al.(2017)
documents that ETFs are linked to wider bid-ask spreads and higher Amihud
illiquidity measure - a lower liquidity. On the other hand, Hegde and Mc-
Dermott(2003), Schoenfeld(2017) find that S&P500 additions lead to sustained
increase in the liquidity of the added stocks. Saglam et al(2020) finds that
increases in ETF ownership are associated with higher liquidity. Liebi(2020)
and Kitajimaa(2022) find that stocks owned more by index funds have lower
transaction costs hence higher liquidity. However, Saglam et al(2020) also doc-
ument that stocks with high ETF ownership may experience impaired liquidity
during major market stress events. My current work adds empirical evidence
to the stream of literature that higher index ownership leads to a more liquid
market. And my future extension to panel regression with longer time horizon
can testify whether the impact of index ownership on liquidity reverts during
financial crisis like COVID-19 recession, 2020 stock market crash and add more
evidence to Saglam’s findings.
6


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


<!-- Page 10 -->

bid-ask spread( BAS i,q), trading volume( TVi,q), shares turnover( TVi,q), Ami-
hub illiquidity measure( ILLIQ i,q), idiosyncratic volatility( V OL i,q). Controls
inXi,qinclude one quarter lagged log(price), one quarter lagged market capi-
talization, lagged returns from quarter q-3 to q-1 and institutional ownership.
The controls in Xi,qare selected to capture stock characteristics correlated with
index ownership and to reduce the potential impact of autocorrelation issues in
financial time series.
4.2 Variable Construction
4.2.1 Stock Return Synchronicity
Stock return synchronicity reflects the degree of how much the stock returns
is explained by the returns from the market index. Following the prior studies
(Dasgupta et al.(2010), Ferreira et al.(2011); Devos et al.(2015)), I first estimate
R2from the Fama-French three-factor model. In a quarter, I regress the daily
excess return ( Rit−Rft) on the market excess return ( RMt−Rft), size premium
(SMB t), and value premium ( HML t):
Rit−Rft=αit+β1(RMt−Rft) +β2SMB t+β3HML t+ϵit
Similar to prior works on stock return synchronicity (e.g., Morck et al.(2000);
Chan and Chan(2014); Kan and Gong(2018)), I use a logistic transformation
ofR2to define the stock return synchronicity( SY NCH i,q) for stock iin the
quarter qas:
SYNCH i,q= ln 
R2
i,q
1−R2
i,q!
(2)
4.2.2 Liquidity and Volatility
The liqudity measures I use for analysis including bid-ask spread( BAS i,q):
BAS i,q=Aski,q−Bidi,q
Midpoint i,q∗100 (3)
9


<!-- Page 11 -->

where BAS i,qis the simplest type of bid-ask spread - the quoted spread. A
higher BAS i,qreflects lower liquidity as the gap between demand and supply of
liquidity is widened.
Trading volume( TVi,q) is the total number of shares transacted for stock iin
quarter q. Shares turnover( TVi,q) is the total number of trading shares divided
by the average number of shares outstanding for the same period.
Amihub illiquidity measure( ILLIQ i,q) is one of the most commonly used mea-
sure for stock illiquidity in the finance literature, proposed by Amihud(2002):
ILLIQ( i, q) =1
nnX
t=1|Ri,t|
Pi,t∗vi,t(4)
where nis the total trading days in quarter q.
Idiosyncratic volatility( V OL i,q) is the standard deviation of stock i’s return
in quarter q. It is constructed from the daily close prices and converted into
quarter frequency.
V OL i,q=σi,t∗√n (5)
4.2.3 Index Ownership
I define the index ownership as the fraction of a stock’s shares held by index
funds and ETFs out of its total shares outstanding. I select the funds by their
names identified as index funds and ETFs and use the holding data from Thom-
son S12 to construct the index ownership( IND i,q):
IND i,q=shares of stock iheld by index funds and ETFs in quarter q
total shares outstanding in quarter q(6)
4.3 Descriptive Statistics
Table 1 describes the summary statistics of the sample stocks. I report the mean
and median figures as well as the figures at different quantiles. In general, the
distribution of the stock characteristic variables is highly skewed to the right as
the mean is much larger than the median. Except for the SY NCH , it is more
10


<!-- Page 12 -->

symmetrical, with a small difference between mean and median values. Index
and institutional ownership display a relatively pretty or symmetric distribution
because I use Interquartile Range(IQR) method to identify and drop outliers in
these observations.
5 Regression Analysis
The regression results are displayed in Table 2. Column 1 shows a positive
correlation between index ownership and stock return synchronicity. As higher
SY NCH indicates higher price informativeness. This tells us that larger index
ownership makes the market more transparent. Column 2 and 5 show that bid-
ask spread and Amihud illiquidity measure are negatively correlated with index
ownership. Since tighter spreads reflect better liquidity, higher index ownership
brings more liquidity to the market. It is surprising that column 3 and 4 report
opposite results. And it requires deeper examination to find the underlying
logic. The last column shows that volatility is positively correlated with index
ownership. Although the baseline regression result for V OL is statistically in-
significant, it becomes statistically significant after removing the one-quarter
lagged market capitalization control variable as reported in the last row.
11


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


<!-- Page 15 -->

current data is comprehensive enough to conduct the panel regression.
References
The shift from active to passive investing: Potential risks to financial stabil-
ity? Finance and Economics Discussion Series , 2020. ISSN 19362854. doi:
10.17016/FEDS.2018.060r1.
P. Abedifar, K. Bouslah, and Y. Zheng. Stock price synchronicity and
price informativeness: Evidence from a regulatory change in the u.s. bank-
ing industry. Finance Research Letters , 2020. ISSN 15446123. doi:
10.1016/j.frl.2020.101678.
Y. Amihud. Illiquidity and stock returns: cross-section and time-series effects.
Journal of Financial Markets 5 , pages 31–56, 2002.
J. Bai, T. Philippon, and A. Savov. Have financial markets become more infor-
mative? Journal of Financial Economics 122 , page 625–654, 2016.
I. Ben-David, F. Franzoni, and R. Moussawi. Do etfs increase volatility? The
Journal of Finance 73(6) , pages 2471–2535, 2018.
B. Bennett, R. M. Stulz, and Z. Wang. Does joining the sp 500 index hurt
firms? NBER Working Paper , 2022.
P. Bond and D. Garc´ ıa. The equilibrium consequences of indexing. Working
Paper , 2019.
J. Brogaard, H. Nguyen, T. J. Putnins, and E. Wu. What moves stock prices?
the role of news, noise, and information. Working Paper , 2019.
G. Chabakauri and O. Rytchkov. Asset pricing with index investing. Journal
of Financial Economics 141(1) , pages 195–216, 2021.
K. Chan and Y. C. Chan. Price informativeness and stock return syn-
chronicity: Evidence from the pricing of seasoned equity offerings. Jour-
nal of Financial Economics , 114:36–53, 2014. ISSN 0304405X. doi:
10.1016/j.jfineco.2014.07.002.
14


<!-- Page 16 -->

J. L. Coles, D. Heath, and M. C. Ringgenberg. On index investing. Working
Paper , 2020.
L. W. Cong and D. X. Xu. Rise of factor investing: Asset prices, informational
efficiency, and security design. Working Paper , 2018.
S. Dasgupta, J. Gan, and N. Gao. Transparency, price informativeness,
and stock return synchronicity: Theory and evidence. Journal of Finan-
cial and Quantitative Analysis , 45:1189–1220, 2010. ISSN 00221090. doi:
10.1017/S0022109010000505.
E. Devos, W. Hao, A. K. Prevost, and U. Wongchoti. Stock return syn-
chronicity and the market response to analyst recommendation revisions.
Journal of Banking and Finance , 58:376–389, 2015. ISSN 03784266. doi:
10.1016/j.jbankfin.2015.04.021.
Y. Dong, O. Z. Li, Y. Lin, and C. Ni. Does information processing cost affect
firm-specific information acquisition?-evidence from xbrl adoption. Journal
of Financial and Quantitative Analysis 51(2) , page 435–462, 2016.
E. D´ avila and C. Parlatore. Identifying price informativeness. Working Paper ,
2020.
D. Ferreira, M. A. Ferreira, and C. C. Raposo. Board structure and price
informativeness. Journal of Financial Economics , 99:523–545, 2011. ISSN
0304405X. doi: 10.1016/j.jfineco.2010.10.007.
S. Glossner. Russell index reconstitutions, institutional investors, and corporate
social responsibility. CFR (draft) , pages 1–43, 2020.
L. Glosten, S. Nallareddy, and Y. Zou. Etf activity and informational efficiency
of underlying securities. Working Paper , 2019.
S. J. Grossman and J. E. Stiglitz. On the impossibility of informationally effi-
cient markets. The American Economic Review 70(3) , pages 393–408, 1980.
V. Gr´ egoire. The rise of passive investing and index-linked comovement. Fi-
nancial Analysts Journal 51 , pages 1–16, 2019.
S. J. W. Hamm. The effect of etfs on stock liquidity. 2014.
15


<!-- Page 17 -->

S. P. Hegde and J. B. Mcdermott. The liquidity effects of revisions to the sp
500 index: an empirical analysis. Journal of Financial Markets(6) , pages
413–459, 2003.
C. W. Holden. Index arbitrage as cross-sectional market making. Journal of
Future Markets 15(4) , page 423–455, 1995.
K. Hou, L. Peng, and W. Xiong. Is r 2 a measure of market ine ¢ciency? 2013.
D. Israeli, C. M. C. Lee, and S. A. Sridharan. Is there a dark side to exchange
traded funds (etfs)? an information perspective by. Review of Accounting
Studies 22(3) , page 1048–1083, 2017.
S. Kan and S. Gong. Does high stock return synchronicity indicate high or
low price informativeness? evidence from a regulatory experiment. Inter-
national Review of Finance , 18:523–546, 12 2018. ISSN 14682443. doi:
10.1111/irfi.12157.
J. W. Kim, J.-H. Lim, and W. G. No. The effect of first wave mandatory xbrl re-
porting across the financial information environment. Journal of Information
Systems 26(1) , page 127–153, 2012.
K. Kitajima. Passive investors and concentration of intraday liquidity: Evidence
from the tokyo stock exchange. Pacific Basin Finance Journal , 74, 2022. ISSN
0927538X. doi: 10.1016/j.pacfin.2022.101812.
J. Lee. Passive investing and price efficiency. Working Paper , 2020.
L. J. Liebi. The effect of etfs on financial markets: a literature review. Financial
Markets and Portfolio Management , 34:165–178, 2020. ISSN 23738529. doi:
10.1007/s11408-020-00349-1.
H. Liu and Y. Wang. Index investing and price discovery. Working Paper , 2018.
G. Malikov. Information, participation, and passive investing. Working Paper ,
2019.
W. Y. Randall Morck, Bernard Yeung. The information content of stock mar-
kets: why do emerging markets have synchronous stock price movements?
Journal of Financial Economics 58 , pages 215–260, 2000.
16


<!-- Page 18 -->

G. D. Rossi and M. Steliaros. The shift from active to passive and its effect on
intraday stock dynamics. Journal of Banking and Finance , 143, 2022. ISSN
03784266. doi: 10.1016/j.jbankfin.2022.106595.
M. Saglam, T. Tuzun, and R. Wermers. Do etfs increase liquidity? 2020.
M. Sammon. Passive ownership and price informativeness. Working Paper ,
2022.
J. Schoenfeld. The effect of voluntary disclosure on stock liquidity: New evidence
from index funds. Journal of Accounting and Economics , 63:51–74, 2017.
ISSN 01654101. doi: 10.1016/j.jacceco.2016.10.007.
R. N. Sullivan and J. X. Xiong. How index trading increases market vulnera-
bility. Financial Analysts Journal 68(2) , pages 70–84, 2012.
A. B. S. Sundaresan. More risk, more information: How passive ownership can
improve informational efficiency. Working Paper , 2020.
E. L. Valentin Haddad, Paul Huebner. How competitive is the stock market?
theory, evidence from portfolios, and implications for the rise of passive in-
vesting. 2022.
17