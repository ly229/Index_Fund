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