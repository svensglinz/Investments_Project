# Investments_Project

#load libraries
import yfinance as yf
import pandas as pd
import numpy as np
import math as m
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.optimize import Bounds
from datetime import date

#specify parameters

#need to clean NA's as well...

#how do get dividends from single companies - Do we add dividends to fund's performance?
#get balance sheet info etc.
NOV = yf.Ticker("HOLN.SW")

#get the currency of the stock
NOV.info["currency"]

#industry of company
NOV.info["industry"]

#location of company
NOV.info["country"]
start_date = "2021-01-01"
end_date = date.today()
tickers = ["NOVN.SW", "ROG.SW", "NESN.SW", "ZURN.SW", "CFR.SW"]

stocks = []
earnings = []
dividends = []

for i in tickers:
    yf.Ticker(i).earnings

#comparison index / Benchmark
#Benchmark_Ticker =
#Portfolio_Tickers =
#E_Rate_Tickers =

Benchmark = yf.download(tickers = "^SSMI", start = start_date, end = end_date, interval = "1d")
data = yf.download(tickers = tickers, start = start_date, end = end_date, interval = "1d")
E_Rate = yf.download(tickers = ["EUR=X", "CHF=X"], start = start_date, end = end_date, interval = "1d")

Benchmark = Benchmark["Adj Close"]
data = data["Adj Close"]
E_Rate = E_Rate["Adj Close"]

Benchmark.plot()

#clean data (replace by existing value before - do not remove value)
data.isnull().sum()

cut_off_date = "2021-12-31"
data_in_sample = data[data.index <= cut_off_date]
data_out_sample = data[data.index > cut_off_date]

#visualize indexed performance of all stocks
data_vis = data.div(data.iloc[0,:])
plt.plot(data_vis)
plt.legend(data_vis.columns)

#estimate expected returns
ER = data.pct_change().mean()
S = data.pct_change().cov()

#estimate needed imputs for the model
def pvar(w, S):
return (w.T @ S @ w)

def pret(w, ER):
return (w.T @ ER)

def sharpe(w, ER, S):
return -(w.T @ ER)/ ((w.T @ S @ w) ** 0.5)

N = len(ER)
x0 = np.ones(N)/N

cons = ({"type": "eq", "fun" : lambda x: np.sum(x) - 1})
bounds = Bounds(-1/N*10, 1/N*10)
GMVP = minimize(pvar, x0, method='SLSQP', args=S, constraints=cons, options={'disp': True, 'ftol': 1e-9})
MSRP = minimize(sharpe, x0, method='SLSQP', args=(ER,S), constraints=cons, options={'disp': True, 'ftol': 1e-9}, bounds = bounds)

#out of sample performance
performance_in_sample = data_in_sample * MSRP.x
performance_in_sample = performance_in_sample.sum(axis = 1)
performance_in_sample = performance_in_sample.div(performance_in_sample[0])

performance_out_sample = data_out_sample * MSRP.x
performance_out_sample = performance_out_sample.sum(axis = 1)
performance_out_sample = performance_out_sample.div(performance_out_sample[0])

performance_sample = data * MSRP.x
performance_sample = performance_sample.sum(axis = 1)
performance_sample = performance_sample.div(performance_sample[0])

performance_benchmark = Benchmark.div(Benchmark[0])
plt.plot(performance_in_sample)
plt.plot(performance_benchmark)
plt.plot(performance_in_sample[len(performance_in_sample)-1]-1 + performance_out_sample)
