#load libraries
#---------------------------------------------------------------------------
import yfinance as yf
import pandas as pd
import numpy as np
import math as m
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.optimize import Bounds
import datetime as dt

#define parameters of investment period start
start_backtesting = np.datetime64("2011-01-01")
end_backtesting = np.datetime64("2021-12-31")

#import needed files
#-----------------------------------------------------------------------------
constituents = pd.read_csv("index_constituents_data.csv", index_col= 0)
benchmark = pd.read_csv("benchmark.csv", index_col= 0)
gross_returns = pd.read_csv("Gross_Prices_EUR.csv", index_col= 0)
net_returns = pd.read_csv("Net_Prices_EUR.csv", index_col= 0)

#select stocks to invest in (10 highest dividend yields and 10 lowest dividend yields)
constituents = constituents.sort_values(by = "Yield", ascending= False)
stocks = constituents.iloc[np.r_[0:10, 40:50]]

#retreive in sample stock prices to calculate return and variances for optimization
Stock_Prices = Gross_Price[stocks.index.to_list()]
in_sample = Stock_Prices[Stock_Prices.index < end_backtesting]

#estimate expected returns and variance-covariances
ER = in_sample.pct_change().mean()
S = in_sample.pct_change().cov()

#define functions for optimization
#------------------------------------------------------------------------------
def pvar(w, S):
    return (w.T @ S @ w)

def pret(w, ER):
    return (w.T @ ER)

def sharpe(w ,ER, S):
    return -(w.T @ ER)/ ((w.T @ S @ w) ** 0.5)

#calculate optimized portfolios
#---------------------------------------------------------------------------------------
N = len(ER)
x0 = np.ones(N)/N

#set up constraints
#first contraint -> total investment = 100% long
#second constraint- > Values smaller than - min_weight
#third constraint -> Values larger than min_weight,

min_weight = 0.01

cons = ({"type": "eq", "fun" : lambda x: np.sum(x) - 1},
        {"type": "ineq", "fun" : lambda x: -min_weight -x[10:20]},
        {"type": "ineq", "fun" : lambda x: x[0:10]- min_weight})

#define maximum short and long % for each stock
bounds = Bounds(-1/N*4, 1/N*4)

#calculate optimized values (GMVP and MSRP constrained and unconstrained)
GMVP = minimize(pvar, x0, method='SLSQP', args=S, options={'disp': True, 'ftol': 1e-9})
MSRP_const = minimize(sharpe, x0, method='SLSQP', args=(ER, S), constraints=cons, options={'disp': True, 'ftol': 1e-9}, bounds = bounds)
MSRP_unconst = minimize(sharpe, x0, method='SLSQP', args=(ER,S), constraints=cons, options={'disp': True, 'ftol': 1e-9})

#complete data set of selected stocks with weights
stocks["weights"] = MSRP_const.x

#calculate strategy returns in and out of sample
#-------------------------------------------------------------------------------

benchmark.index = pd.to_datetime(benchmark.index)

#out of sample returns (net)
out_sample_net = Net_Price[Net_Price.index > end_backtesting]
out_sample_net = out_sample_net[stocks.index] * MSRP_const.x
out_sample_net =  out_sample_net.sum(axis = 1)

#out of sample returns (gross)
out_sample_gross = Gross_Price[Gross_Price.index > end_backtesting]
out_sample_gross = out_sample_gross[stocks.index] * MSRP_const.x
out_sample_gross =  out_sample_gross.sum(axis = 1)

#in sample returns (net)
in_sample_net = Net_Price[Net_Price.index <= end_backtesting]
in_sample_net = in_sample_net[stocks.index] * MSRP_const.x
in_sample_net =  in_sample_net.sum(axis = 1)

#in sample returns (gross)
in_sample_gross = Gross_Price[Gross_Price.index <= end_backtesting]
in_sample_gross = in_sample_gross[stocks.index] * MSRP_const.x
in_sample_gross =  in_sample_gross.sum(axis = 1)

#combine net and gross in sample weighted value to data frame & left join index values (net and gross)
#then calculate relative performance of all series
in_sample = pd.DataFrame({"strategy_net": in_sample_net, "strategy_gross": in_sample_gross},
                         index = in_sample_net.index)
in_sample = in_sample.join(benchmark)
in_sample = in_sample.div(in_sample.iloc[0])

#combine net and gross out of sample weighted value to data frame & left join index values (net and gross)
#then calculate relative performance of all series
out_sample = pd.DataFrame({"strategy_net": out_sample_net, "strategy_gross": out_sample_gross},
                         index = out_sample_net.index)
out_sample = out_sample.join(benchmark)
out_sample = out_sample.div(out_sample.iloc[0])

#plot results of in and ot of sample performances
#-------------------------------------------------------------------------------

#out of sample strategy vs. benchmark
plt.plot(out_sample[["strategy_gross", "benchmark_gross"]])
plt.legend(out_sample[["strategy_gross", "benchmark_gross"]].columns)
plt.savefig("out_sample.png", device = "png")

#in sample strategy vs. benchmark
plt.plot(in_sample[["strategy_gross", "benchmark_gross"]])
plt.legend(in_sample[["strategy_gross", "benchmark_gross"]].columns)
plt.savefig("in_sample.png", device = "png")

#show strategy returns net vs. gross!
plt.plot(in_sample[["strategy_gross", "strategy_net"]])
plt.legend(in_sample[["strategy_gross", "strategy_net"]].columns)
plt.savefig("net_vs_gross.png", device = "png")

#calculate and plot portfolio characteristics
#------------------------------------------------------------------------------------
#plot weights for different optimization techniques (GMVP, MSRP (constrained and unconstrained)
#...
#...
#...

#plot weights diagram
width = 0.4
fig, ax = plt.subplots()
ind = np.arange(len(stocks))

ax.barh(ind, stocks.weights, width, label = "Strategy Weights")
ax.barh(ind + width, stocks.index_weights, width, label = "Index Weights")
ax.set(yticks = ind + width, yticklabels = stocks.Name)
ax.legend()
plt.title("Stock Weights Strategy vs Index", size = 20)
plt.show()
plt.savefig("weights.png", device = "png")

country_weights = stocks[["Country", "weights"]].groupby(by = "Country").sum()

#plot currencies
currency_weights_strategy = stocks[["Currency", "weights"]].groupby(by = "Currency").sum()
currency_weights_index = constituents[["Currency", "index_weights"]].groupby(by = "Currency").sum()

plt.pie(currency_weights_strategy.weights, labels= currency_weights_strategy.index, autopct='%1.1f%%')
plt.savefig("currency_pie_strategy.png", device = "png")

#plot weights diagram index vs stragegy
currency_weights = currency_weights_index.join(currency_weights_strategy)

width = 0.4
fig, ax = plt.subplots()
ind = np.arange(len(currency_weights_index))

ax.barh(ind, currency_weights.index_weights, width, label = "Strategy Weights")
ax.barh(ind + width, currency_weights.weights, width, label = "Index Weights")
ax.set(yticks = ind + width, yticklabels = currency_weights.index)
ax.legend()
plt.title("Currency Weights Strategy vs Index", size = 20)
plt.show()
plt.savefig("curency_comparison.png", device = "png")

#plot industries
sector_weights_strategy = stocks[["Sector", "weights"]].groupby(by = "Sector").sum()
sector_weights_index = constituents[["Sector", "index_weights"]].groupby(by = "Sector").sum()

sector_weights = sector_weights_index.join(sector_weights_strategy)

width = 0.4
fig, ax = plt.subplots()
ind = np.arange(len(sector_weights))

ax.barh(ind, sector_weights.index_weights, width, label = "Strategy Weights")
ax.barh(ind + width, sector_weights.weights, width, label = "Index Weights")
ax.set(yticks = ind + width, yticklabels = sector_weights.index)
ax.legend()
plt.title("Sector Weights Strategy vs Index", size = 20)
plt.show()
plt.savefig("sector_weights.png", device = "png")

#calculate performance ratios (in and out of sample!!!)
#-------------------------------------------------------------------------------------------
rf_rate = -0.5
SR_in_sample = sharpe(w = MSRP.x, ER = , S = )
SR_benchmark_in_sample = sharpe(w = , ER =  ,S = )

SR_out_sample = sharpe(w = , ER =  ,S = )
SR_benchmark_out_sample = sharpe(w = , ER =  ,S = )

#calculate ratios
stocks_short = stocks[stocks.weights <0]
stocks_long = stocks[stocks.weights > 0]

#dividend yield
div_yield_long = (stocks_long.weights / stocks_long.weights.sum() * stocks_long.Yield).sum()
div_yield_short = (stocks_short.weights.abs()/ stocks_short.weights.abs().sum() * stocks_short.Yield).sum()
div_yield_index = (constituents.index_weights * constituents.Yield).sum()

index = constituents.join(stocks[["weights"]])

#trailing PE Ratio, #forward PE
index = index[index.Trailing_PE.notna()]
index = index[index.Forward_PE.notna()]
index.Trailing_PE = np.where(index.Trailing_PE > 60, 60, index.Trailing_PE)
index.Forward_PE = np.where(index.Forward_PE > 60, 60, index.Forward_PE)
index.PB_Ratio =  np.where(index.PB_Ratio > 25, 25, index.PB_Ratio)

short = index.loc[stocks_short.index]
long = index.loc[stocks_long.index]

PE_long = (long.weights / long.weights.sum() * long.Trailing_PE).sum()
PE_short = (short.weights / short.weights.sum() * short.Trailing_PE).sum()
PE_index = (index.index_weights * index.Trailing_PE).sum()

PE_fwd_long = (long.weights / long.weights.sum() * long.Forward_PE).sum()
PE_fwd_short = (short.weights / short.weights.sum() * short.Forward_PE).sum()
PE_fwd_index = (index.index_weights * index.Forward_PE).sum()
#PB Ratio
PB_long = (long.weights / long.weights.sum() * long.PB_Ratio).sum()
PB_short = (short.weights / short.weights.sum() * short.PB_Ratio).sum()
PB_index = (index.index_weights * index.PB_Ratio).sum()

#assemble metrics dataframe
ratios_short = {"Yield": div_yield_short, "Price_Book": PB_short, "Trailing_PE": PE_short,
                "Forward_PE": PE_fwd_short}
ratios_long = {"Yield": div_yield_long, "Price_Book": PB_long, "Trailing_PE": PE_long,
                "Forward_PE": PE_fwd_long}

ratios_index = {"Yield": div_yield_index, "Price_Book": PB_index, "Trailing_PE": PE_index,
                "Forward_PE": PE_fwd_index}

ratios_table = pd.DataFrame({"Portfolio Short":ratios_short, "Portfolio Long": ratios_long, "Index": ratios_index})

#explain out out sample outperformance!
e_rates = yf.download(tickers = ["CHFEUR=X", "GBPEUR=X"], start = end_backtesting, end = dt.date.today(), interval = "1d")
e_rates = e_rates["Adj Close"]
e_rates = e_rates.rename(columns= {"CHFEUR=X": "CHF", "GBPEUR=X": "GBP"})
e_rates_rel = e_rates.div(e_rates.iloc[0])
e_rates_rel.plot()

gross_returns.index = pd.to_datetime(gross_returns.index)
gross_returns_rel = gross_returns[gross_returns.index > end_backtesting]
gross_returns_rel = gross_returns_rel.loc[:,stocks.index]
gross_returns_rel = gross_returns_rel.div(gross_returns_rel.iloc[0])

gross_returns_rel.plot()