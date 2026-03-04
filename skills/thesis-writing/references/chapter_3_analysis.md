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