"""
This file assembles a data frame which contains all stocks from the stoxx europe 50 index
and includes various stock ratios such as PE/ PB ratios, dividend yield as of 31.12.2021 etc.
Further, we download the prices of all index constituents and the index itself
and modify them to get the gross values (incl. dividends)
We export all prices as csv files which can be used as inputs for the script "calculation.py"
"""

# load libraries
import yfinance as yf
import pandas as pd
import numpy as np
import time
from yahoo_fin.stock_info import *
import random

############################################################################
# define parameters of investment period start
############################################################################

#start_backtesting is the minimum value. As it turns out, one of the stocks in our portfolio
#only IPO'd in 2019 and thus, our in-sampel period is accordingly cut short.

start_backtesting = np.datetime64("2011-01-01")
end_backtesting = np.datetime64("2021-12-31")
end_out_sample = np.datetime64("2022-05-21")

# define end backtesting - one year as interval to calculate one year dividend yield
end_dividends = np.datetime64("2021-12-31")
start_dividends = end_dividends - np.timedelta64(365, 'D')

############################################################################
# Import benchmark and calculate net and gross values
############################################################################

"""
ishares ETF on Stoxx Europe 50 used as benchmark (ETF is distributing --> paid out dividends were downloaded
separately and are stored in file "dividends_benchmark.csv" which is needed to calculate gross returns
with reinvested dividends)
"""

benchmark = yf.download(tickers="EUNA.AS", interval="1d", start=start_backtesting, end=end_out_sample)
benchmark_dividends = pd.read_csv("files/dividends_benchmark.csv")
benchmark_dividends.index = pd.to_datetime(benchmark_dividends.Date, format="%d.%m.%Y")
benchmark_dividends = benchmark_dividends.drop("Date", axis=1)

benchmark_net = pd.DataFrame(benchmark["Adj Close"])

# calculate gross index with immediate reinvesting of all dividends
benchmark_gross = benchmark_net.join(benchmark_dividends)
benchmark_gross["share_purchased"] = benchmark_gross.Dividend / benchmark_gross["Adj Close"]
benchmark_gross.share_purchased = benchmark_gross.share_purchased.fillna(0)
benchmark_gross.share_purchased = benchmark_gross.share_purchased.cumsum()
benchmark_gross.share_purchased = benchmark_gross.share_purchased + 1
benchmark_gross = benchmark_gross["Adj Close"] * benchmark_gross.share_purchased

benchmark_gross = pd.DataFrame({"Adj Close": benchmark_gross})

# write CSV for replicability
benchmark = pd.DataFrame({"benchmark_gross": benchmark_gross["Adj Close"], "benchmark_net": benchmark_net["Adj Close"]},
                         index=benchmark_net.index)

benchmark.to_csv("files/benchmark.csv")

# --------------------------------------------------------------------------------------------------
"""
Download ticker symbols for investment universe which is the stoxx europe 50 index.
as there are no ticker symbols available on the webpage do the following:
1: download names of index constituents from wikipedia
2: download names and ticker of euro stoxx 50 from wikipedia
3: join the two since many constituents are overlapping
4 check which tickers are still missing and manually complete them!

!!! Warning: Running this code in the future could produce errors if the information on the pages that we scrape 
changes
"""

# Retreive Stoxx Europe 50 Data
page_StoxxEurope = pd.read_html("https://de.wikipedia.org/wiki/STOXX_Europe_50")
StoxxEurope_table = page_StoxxEurope[4][["Name"]]

# Retrieve Estoxx 50 Data
page_Estoxx = pd.read_html('https://en.wikipedia.org/wiki/EURO_STOXX_50')
Estoxx_table = page_Estoxx[3]
tickers_Estoxx = Estoxx_table[["Ticker", "Name"]]

# complete missing tickers manually
temp = StoxxEurope_table.merge(Estoxx_table[["Ticker", "Name"]], on="Name", how="left")
missing_names = temp[temp["Ticker"].isna()].Name
missing_names = missing_names.to_list()
missing_tickers = ["ABBN.SW", "ASML.AS", "AZN.L", "BHP.L", "BP.L",
                   "BATS.L", "MBG.DE", "DGE.L", "GSK.L", "HSBA.L", "LOR.F", "LIN.DE",
                   "MOH.F", "NG.L", "NESN.SW", "NOVN.SW", "NOVO-B.CO", "PRU.L", "RKT.L",
                   "REL.L", "RIO.L", "ROG.SW", "SHEL.L", "UBSG.SW", "ULVR.L",
                   "DG.PA", "VOD.L", "ZURN.SW"]

missing_df = {"Name": missing_names, "Ticker": missing_tickers}
missing_df = pd.DataFrame(missing_df)

# assemble final data frame which contains names and tickers of all stoxx europe 50 index members
all_tickers = tickers_Estoxx.append(missing_df)
Tickers_StoxxEurope = StoxxEurope_table.merge(all_tickers, on="Name", how="left")

Stock_Tickers = Tickers_StoxxEurope.Ticker.to_list()

############################################################################
# retrive financial ratios and calculate one year dividend yield as of end-backtesting date for all index tickers
############################################################################

# initilize empty lists to store results
ticker_list = []
div_yield_list = []
currency_list = []
sector_list = []
pb_list = []
forward_pe_list = []
trailing_pe_list = []
country_list = []
dividend_dict = {}

"""
unfortunately accessing yfinance via api (yfinance package) does not always
return dividends or ratio information even if they exist.
Thus, for ratios and dividends we rely on yahoo_fin module which is much more reliable.
However, since it scrapes the data directly from the website and yahoo has very strict
rate limits, we pause the loop for a random time(between 100 & 200 seconds) after the website returns that we cannot
fetch any more data.
We notice that we are blocked by the website if: 
dividend data frame comes back empty --> indexerror is raised by loop 
ratios data frame with nan is produced for ratios -> We manually raise an index error to indicate
that the rate limit has been reached
"""

"""
!!! Warning: 
This loop can take quite a long time to complete!
If you wish to skip this step, please continue below and just import the results 
of this loop as a csv
- Also, the ratios which we download (eg. PB / PE ratios are the ratios from the day the script is run. Therefore 
running the script at different times could produce slightly different results as unfortunately, we were 
only able to get the most recent ratios and not the ratios per a specific date.
"""

i = 0
while i in range(len(Stock_Tickers)):

    try:
        ticker = yf.Ticker(Stock_Tickers[i])
        info = ticker.info
        dividends = get_dividends(Stock_Tickers[i])

        if dividends.empty:
            div_yield = 0
        else:
            # calculate dividend yield as of 31. December 2021
            dividends = dividends.drop("ticker", axis=1)
            dividends_year = dividends.iloc[(dividends.index >= start_dividends) & (dividends.index < end_dividends)]
            dividends_year = dividends_year.sum()
            price = ticker.history(start=end_backtesting - 10, end=end_backtesting)
            price = price.Close
            price = price[-1]
            div_yield = (dividends_year / price)[0]

        dividend_dict[Stock_Tickers[i]] = dividends

        # get ratios
        ratios = get_stats_valuation(Stock_Tickers[i])
        ratios = ratios.rename(columns={0: "ratio", 1: "value"})
        ratios.index = ratios.ratio
        ratios = ratios.drop("ratio", axis=1)

        # if limit is reached, yahoo returns all NA's for ratios...
        if ratios.value.isnull().all():
            raise IndexError

        # store raios and informational values in respective lists
        currency_list.append(info.get("currency"))
        sector_list.append(info.get("sector"))
        pb_list.append(ratios.loc[ratios.index == "Price/Book (mrq)"].value[0])
        forward_pe_list.append(ratios.loc[ratios.index == "Forward P/E"].value[0])
        trailing_pe_list.append(ratios.loc[ratios.index == "Trailing P/E"].value[0])
        country_list.append(info.get("country"))
        div_yield_list.append(div_yield)

        # print progress of loop
        print("import of " + Stock_Tickers[i] + " successful")
        i = i + 1
        time.sleep(random.randint(3, 15))

    # loop is paused after rate limit has been reached
    except IndexError:
        pause = random.randint(100, 200)
        print("pause for " + str(pause) + " seconds")
        time.sleep(pause)
        continue

# assemble as data.frame
stocks = pd.DataFrame({"Name": list(Tickers_StoxxEurope["Name"]), "Currency": currency_list, "Country": country_list,
                       "Sector": sector_list, "Yield": div_yield_list, "Forward_PE": forward_pe_list,
                       "Trailing_PE": trailing_pe_list, "PB_Ratio": pb_list},
                      index=Stock_Tickers)

"""
#to compare our strategy with that of the index, we also manually add the weights
as found on the ishares website to it (per 31. December 2021). 
4 index members have been replaced since and thus there is a difference between the ishares data
and our index constituents (Vodafone, Safran, National Grid, BHP)
for each, the weight of its replacement is taken. The Impact of this is expected to be very minor as 
all weights are < 2%
"""

stocks_indexweights = np.array([1.19, 0.99, 1.36, 1.52, 1.37, 1.77, 0.99, 6.12,
                                3.35, 1.13, 1.19, 0.96, 1.16, 1.47, 1.63, 1.44, 1.18, 1.18, 2.34,
                                1.14, 2.02, 2.31, 1.27, 0.87, 1.08, 2.12, 3.27, 4.01, 1.1, 7.25, 3.95, 3.43, 1.34, 0.87,
                                1.16,
                                1.19, 1.31, 5.38, 1.66, 2, 2.12, 2.85, 2.06, 2.4, 2.46,
                                1.14, 2.59, 1.16, 0.8, 1.22]) / 100

stocks["index_weights"] = stocks_indexweights

# write data to CSV
stocks.to_csv("files/index_constituents_data.csv")

"""
If you chose not to run the loop above, run the below code to continue
--> CODE: stocks = pd.read_csv("files/index_constituents_data.csv)
"""

############################################################################
# Download Prices of Benchmark constituents and Exchange Rates to convert all Prices to EUR
############################################################################

Net_Price = yf.download(tickers=list(stocks.index), start=start_backtesting, end=end_out_sample, interval="1d")
Net_Price = Net_Price["Adj Close"]

# get unique currencies of stocks
currencies = stocks.Currency.unique()

# download exchange rates against EUR of all currencies which are represented in the index
e_rates = yf.download(tickers=["CHFEUR=X", "DKKEUR=X", "GBPEUR=X"], start=start_backtesting, end=end_out_sample,
                      interval="1d")
e_rates = e_rates["Adj Close"]
e_rates = e_rates.rename(columns={"CHFEUR=X": "CHF", "DKKEUR=X": "DKK", "GBPEUR=X": "GBP"})

# fill values where we do not have data with previous value or delete if it is the first value

Net_Price = pd.DataFrame(Net_Price).fillna(method="ffill")
Net_Price = Net_Price.fillna(0)
Net_Price = Net_Price.loc[np.all(Net_Price != 0, axis=1)]

############################################################################
# Calculate Gross Stock Prices
############################################################################

Gross_Price = pd.DataFrame()

# calculate gross returns based on assumption that each dividend is immediately reinvested in the given stock
for i in Net_Price.columns:

    temp = pd.DataFrame(Net_Price[i])
    # handle stocks for which no entry in dividend_dict exists
    try:
        temp = temp.join(dividend_dict.get(i))
        temp["share_purchased"] = temp.dividend / temp[i]
        temp.share_purchased = temp.share_purchased.fillna(0)
        temp.share_purchased = temp.share_purchased.cumsum()
        temp.share_purchased = temp.share_purchased + 1
        temp = temp[i] * temp.share_purchased

    except AttributeError:
        pass

    Gross_Price[i] = temp

# export gross prices in local currency as csv
Gross_Price.to_csv("files/Gross_Prices_localccy.csv")

# convert all prices to EUR! (gross and net)
adj_prices_gross = Gross_Price.join(e_rates)
adj_prices_net = Net_Price.join(e_rates)

for i in currencies:
    tickers = stocks.index[stocks.Currency == i]

    # for stocks quoted in pence (100th of a Pound)
    if i == "GBp":
        adj_prices_gross[tickers] = adj_prices_gross[tickers].multiply((adj_prices_gross["GBP"] / 100), axis=0)
        adj_prices_net[tickers] = adj_prices_net[tickers].multiply((adj_prices_net["GBP"] / 100), axis=0)
    elif i == "EUR":
        pass
    else:
        adj_prices_gross[tickers] = adj_prices_gross[tickers].multiply(adj_prices_gross[i], axis=0)
        adj_prices_net[tickers] = adj_prices_net[tickers].multiply(adj_prices_net[i], axis=0)

# delete exchange rates again
adj_prices_net = adj_prices_net[Gross_Price.columns]
adj_prices_gross = adj_prices_gross[Gross_Price.columns]

# write to csv files
adj_prices_net.to_csv("files/Net_Prices_EUR.csv")
adj_prices_gross.to_csv("files/Gross_Prices_EUR.csv")
