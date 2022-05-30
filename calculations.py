"""
The following script has 4 main sections.
We first calculate the optimal investment weights of a trading strategy
by optimization. Then, we calculate the in and out of sample returns
for the strategy and the chosen benchmark.
Ultimately, we calculate performance and risk ratios and plot portfolio
characteristics of the investment strategy in comparison to the benchmark.
 """

#load libraries
import yfinance as yf
import pandas as pd
import numpy as np
import math as m
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.dates as mdates
from scipy.optimize import minimize
from scipy.optimize import Bounds
import datetime as dt
from sklearn.linear_model import LinearRegression
import dataframe_image as dfi

#############################################################################
#                                                                           #
#                                #SECTION 1                                 #
#       Define key parameters, import needed files and define investment   #
#       universe of stocks which the strategy invests in                    #
#                                                                           #
#############################################################################

#define parameters of investment period start (start_backtesting as maximum time period!
#actual time could be shorter depending on availability of data!
start_backtesting = np.datetime64("2011-01-01")
end_backtesting = np.datetime64("2021-12-31")
end_out_sample = np.datetime64("2022-05-21")

#Import needed CSV files
stocks_index = pd.read_csv("files/index_constituents_data.csv", index_col= 0)
benchmark = pd.read_csv("files/benchmark.csv", index_col= 0)
Gross_Price = pd.read_csv("files/Gross_Prices_EUR.csv", index_col= 0)
Net_Price = pd.read_csv("files/Net_Prices_EUR.csv", index_col= 0)

#format index to date
Gross_Price.index = pd.to_datetime(Gross_Price.index)
Net_Price.index = pd.to_datetime(Net_Price.index)
benchmark.index = pd.to_datetime(benchmark.index)

"""
Select stocks which the strategy invests in
our strategy goes long in the top 10 highest dividend stocks and short in the
top 10 lowest dividend stocks --> We select these stocks as our investmnet stocks
drop BHP stock for which dividends have not been calculated correctly!
"""

stocks_index = stocks_index.sort_values(by = "Yield", ascending= False)
stocks_index = stocks_index.drop("BHP.L")
stocks_invest = stocks_index.iloc[np.r_[0:10, 39:49]]

#select prices of stocks which we invest in
Gross_Price_selected = Gross_Price[stocks_invest.index]
Net_Price_selected = Net_Price[stocks_invest.index]

#############################################################################
#                                                                           #
#                                #SECTION 2                                 #
#     Calculate Optimal Strategy stock weights with various constraints     #
#                       and visualize optimization                          #
#                                                                           #
#############################################################################

#estimate expected returns and var-cov matrix for optimization (with in sample data!)

rf_daily = 0
ER = Gross_Price_selected[Gross_Price_selected.index < end_backtesting].pct_change().mean()
S = Gross_Price_selected[Gross_Price_selected.index < end_backtesting].pct_change().cov()

#define functions for optimization (variance, return,  negative sharp ratio)
def pvar(w, S):
    return (w.T @ S @ w)

def pret(w, ER):
    return (w.T @ ER)

def sharpe(w ,ER, S):
    return -(w.T @ (ER-rf_daily))/ ((w.T @ S @ w) ** 0.5)


# ---------------------------------------------------------------------------
# Calculate Optimized Portfolio
# ---------------------------------------------------------------------------
N = len(ER)
x0 = np.ones(N)/N

"""
#set up constraints
first constraint -> total investment = 100% long
second constraint- > Values smaller than - min_weight (shocks which are shorted)
third constraint -> Values larger than min_weight (stock which are longed)
bounds = maximum long and short % for stocks
"""

min_weight = 0.02

cons_multiple = ({"type": "eq", "fun" : lambda x: np.sum(x) - 1},
                {"type": "ineq", "fun" : lambda x: -min_weight -x[10:20]},
                {"type": "ineq", "fun" : lambda x: x[0:10]- min_weight})

bounds = Bounds(-0.2, 0.2)

#only constraint for unconstrained portfolio --> Total stock weights == 100%
cons_simple = ({"type": "eq", "fun" : lambda x: np.sum(x) - 1})

#calculate optimized values (GMVP and MSRP constrained and unconstrained)
GMVP_const = minimize(pvar, x0, method='SLSQP', args=S, constraints = cons_multiple,
                      options={'disp': True, 'ftol': 1e-9}, bounds = bounds)

GMVP_unconst = minimize(pvar, x0, method = "SLSQP", args = S,constraints = cons_simple,
                        options={'disp': True, 'ftol': 1e-9})

MSRP_const = minimize(sharpe, x0, method='SLSQP', args=(ER, S), constraints=cons_multiple,
                      options={'disp': True, 'ftol': 1e-9}, bounds = bounds)

MSRP_unconst = minimize(sharpe, x0, method='SLSQP', args=(ER,S), constraints = cons_simple,
                        options={'disp': True, 'ftol': 1e-9})


#complete data set of selected stocks with calculated constrained MSRP values & export for presentation
stocks_invest = stocks_invest.assign(weights = MSRP_const.x)
stocks_invest = stocks_invest.assign(weights_unconst = MSRP_unconst.x)
dfi.export(stocks_invest, "plots/selected_portfolio_characteristics.png")

#---------------------------------------------------------------------------
                  #Visualize efficient Frontier
#---------------------------------------------------------------------------

"""
Generate Minimum Variance Frontier / Investment Frontier plot by doing the following: 
1. Minimize the negative value of the expected portfolio return 
given a deterministic variance and the same rules as above (eg. min 2% in each stock)
as restrictions in the optimization. --> We thus retrieve the maximum return given 
all restrictions for a given variance. 

2. We construct "random portfolios" which fulfill all the restrictions by minimizing a function
which consists of a random array and a modulus operation which should randomize the optimization 
results. We can then calculate the expected return and variance of these portfolios and add them 
to the plot.
"""
#calculate volatility and expected return of GMVP and MSRP constrained for plotting further below
GMVP_const_ER = pret(w = GMVP_const.x, ER = ER) * 250
GMVP_const_VAR = m.sqrt(pvar(w = GMVP_const.x, S = S) * 250)
MSRP_const_ER =  pret(w = MSRP_const.x, ER = ER) * 250
MSRP_const_VAR = m.sqrt(pvar(w = MSRP_const.x, S = S) * 250)

#define negative portfolio return function
def pret_sim(w, ER):
    return (-(w.T @ ER))

#variances to loop over
var_list = np.arange(0.1, 0.6, 0.01)
bounds = Bounds(-0.2, 0.2)

#initialize lists to store results in
MVF_var_const = []
MVF_ret_const = []

#loop over variances and maximize expected return given that portfolio variance = given variance
for i in var_list:
    max_var = i
    cons = ({"type": "eq", "fun" : lambda x: np.sum(x) - 1},
            {"type": "eq", "fun" : lambda x: m.sqrt((x.T @ S @ x)* 250) - max_var},
            {"type": "ineq", "fun": lambda x: -min_weight - x[10:20]},
            {"type": "ineq", "fun": lambda x: x[0:10] - min_weight})

    maximized = minimize(pret_sim, x0, method='SLSQP', args= ER,
                         options={'disp': True, 'ftol': 1e-9},
                         constraints=cons, bounds = bounds)

    #only store result if optimization was successful!
    if maximized.success:
        MVF_var_const.append(i)
        MVF_ret_const.append(maximized.fun * -250)
    else:
        continue

#---------------------------------------------------------------------------
                  #Visualize efficient Frontier
#---------------------------------------------------------------------------

random_portfolio = []

#"random" function which when optimized gives "random" portfolios
def rand_funct(x,y):
    return ((x % y)/y).sum()

#weight constraints of individual stocks
cons = ({"type": "eq", "fun" : lambda x: np.sum(x) - 1},
        {"type": "ineq", "fun": lambda x: -min_weight - x[10:20]},
        {"type": "ineq", "fun": lambda x: x[0:10] - min_weight})

#simulate 100 different portfolios which fulfill the constraints
i = 0
np.random.seed(0)

while i < 100:
    y = np.random.uniform(low= 0, high=1, size=N)
    portfolio_sim = minimize(rand_funct, x0, method='SLSQP', args= y,
                        constraints=cons, options={'disp': True, 'ftol': 1e-9},
                        bounds = bounds)
    i = i + 1

    #only store successful optimizations
    if portfolio_sim.success:
        random_portfolio.append(portfolio_sim.x)
    else:
        continue

MVF_randvar_const = []
MVF_randret_const = []

#calculate return and variance (yearly) from calculated "random" portfolios
for i in range(len(random_portfolio)):
    exp_return = pret(w = random_portfolio[i], ER = ER)
    exp_var = pvar(w = random_portfolio[i], S = S)
    exp_return_year = exp_return * 250
    exp_var_year = m.sqrt(exp_var * 250)

    MVF_randvar_const.append(exp_var_year)
    MVF_randret_const.append(exp_return_year)

#plot results
MVF_var_const.insert(0, GMVP_const_VAR)
MVF_ret_const.insert(0, GMVP_const_ER)
CAL_x = np.linspace(0, MSRP_const_VAR + 0.2, 50, endpoint=True)
SR_MSRP = MSRP_const_ER / MSRP_const_VAR
CAL_y = 0 + SR_MSRP*CAL_x

frontier_points = pd.DataFrame({"variance": MVF_var_const, "return": MVF_ret_const})
max_ret_index = frontier_points["return"].idxmax()

fig, ax = plt.subplots(figsize = (15,10))
plt.plot(frontier_points.variance.iloc[0:max_ret_index + 1], frontier_points["return"].iloc[0:max_ret_index + 1])
plt.plot(frontier_points.variance.iloc[max_ret_index: len(frontier_points)],
         frontier_points["return"].iloc[max_ret_index: len(frontier_points)], color = "grey", ls =  "--")
plt.scatter(MVF_randvar_const, MVF_randret_const)
plt.scatter(GMVP_const_VAR, GMVP_const_ER, s = 70)
plt.scatter(MSRP_const_VAR, MSRP_const_ER, s = 70)
plt.plot(CAL_x, CAL_y)
plt.xlim([0,0.4])
plt.ylim([0,0.4])
plt.title("Investment Frontier Simulated", size = 25)
plt.xlabel("Volatility", size = 15)
plt.ylabel("Return", size = 15)
plt.legend(["Minimum Variance Frontier",
            "Investment Frontier",
            "Capital Allocation Line",
            "random portfolios",
            "GMVP",
            "MSRP"],
           prop={'size': 15})
ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
plt.xticks(size = 15)
plt.yticks(size = 15)
plt.savefig("plots/investmentfrontier.png")

#############################################################################
#                                                                           #
#                                #SECTION 2                                 #
#                           Calculate  returns of                           #
#                 strategy and benchmark in and out of sample               #
#                                                                           #
#############################################################################

#function which calculates the indexed performance (start = 1) of various stocks and start investment weights
def indexed_performance(start, prices, weights, end = None):

    #filter desired time range
    if end is None:
        temp = prices[prices.index >= start]
    else:
        temp = prices[(prices.index >= start) & (prices.index < end)]

    #define daily percentage changes + 1
    temp = temp.pct_change() + 1

    #replace first row with weights of each stock --> Initial Investment = 1
    temp.iloc[0] = weights

    #take cumulative return of each stock * initial investment (weight) and sum up horizontaly
    temp = temp.cumprod().sum(axis = 1)
    return(temp)

#calculate weights of short stocks / long stocks  that they sum up to 1
weights_short = stocks_invest.weights[stocks_invest.weights < 0] / stocks_invest.weights[stocks_invest.weights < 0].sum()
weights_long =  stocks_invest.weights[stocks_invest.weights > 0] / stocks_invest.weights[stocks_invest.weights > 0].sum()

#calculate needed indexed performance series
in_sample_net = indexed_performance(start = start_backtesting, end = end_backtesting,
                                    weights = MSRP_const.x, prices = Net_Price_selected)

in_sample_gross = indexed_performance(start = start_backtesting, end = end_backtesting,
                                    weights = MSRP_const.x, prices = Gross_Price_selected)

out_sample_gross_short = indexed_performance(start = end_backtesting, weights = weights_short,
                                             prices = Gross_Price_selected[stocks_invest[stocks_invest.weights < 0].index])

out_sample_gross_long = indexed_performance(start = end_backtesting, weights = weights_long,
                                            prices = Gross_Price_selected[stocks_invest[stocks_invest.weights > 0].index])

out_sample_gross = indexed_performance(start = end_backtesting, weights = MSRP_const.x,
                                       prices = Gross_Price_selected)

#combine all returns to data frame (one for in and one for out of sample performance)
in_sample = pd.DataFrame({"strategy_net": in_sample_net,
                          "strategy_gross": in_sample_gross},
                         index = in_sample_net.index)

out_sample = pd.DataFrame({"strategy_gross": out_sample_gross,
                           "long_gross": out_sample_gross_long,
                           "short_gross": out_sample_gross_short},
                           index = out_sample_gross.index)

#join benchmark data & calculate indexed performance of benchmark
in_sample = in_sample.join(benchmark)
out_sample = out_sample.join(benchmark)

in_sample[["benchmark_gross", "benchmark_net"]] = in_sample[["benchmark_gross", "benchmark_net"]].div(in_sample[["benchmark_gross", "benchmark_net"]].head(1).iloc[0])
out_sample[["benchmark_gross", "benchmark_net"]] = out_sample[["benchmark_gross", "benchmark_net"]].div(out_sample[["benchmark_gross", "benchmark_net"]].head(1).iloc[0])

#save in and out of sample returns
in_sample.to_csv("files/in_sample.csv")
out_sample.to_csv("files/out_sample.csv")

#############################################################################
#                                                                           #
#                            #SECTION 3                                     #
#        #Calculate in and out of sample risk & performance ratios          #
#                                                                           #
#############################################################################

#---------------------------------------------------------------------------
                        #Annualized Sharp Ratio
#---------------------------------------------------------------------------

#days = days per year (assumed 250 here)
#rf_rate = annualized risk free rate
#prices for which ratio should be alculated

def sharp_ratio(price, days, rf_rate):
    ER = price.pct_change().mean()
    SD = price.pct_change().std()
    SR = (ER-rf_rate/days)/SD * m.sqrt(days)

    return round(SR,2)

#---------------------------------------------------------------------------
                        #Yearly volatility
#---------------------------------------------------------------------------

def yearly_vol(price, days, pct = True):
    vol = m.sqrt(price.pct_change().var() * days)

    if pct:
        return format(vol, ".2%")
    else:
        return vol

#---------------------------------------------------------------------------
                        #alpha and beta
#---------------------------------------------------------------------------

#period = period of returns for which regression should be run (eg. "1Y", "1M", "1d")
#kwargs = optional where only alpha, beta, x or y can be returned form function (param!)
#rf_rate = risk free rate for the "period"
def alpha_beta(strategy, benchmark, period, rf_period, pct = True, **kwargs):


    daily_ret_strategy = strategy.pct_change().fillna(0) + 1
    daily_ret_BM = benchmark.pct_change().fillna(0) + 1

    #calculate excess returns for the period over the risk free rate
    period_ret_strategy = daily_ret_strategy.groupby(pd.Grouper(freq=period)).prod() - 1 - rf_period
    period_ret_BM = daily_ret_BM.groupby(pd.Grouper(freq=period)).prod() -1 - rf_period

    model = LinearRegression().fit(period_ret_BM.to_numpy().reshape((-1,1)),
                                    period_ret_strategy.to_numpy().reshape((-1,1)))

    beta = model.coef_[0][0]

    if pct:
        alpha = format(model.intercept_[0], ".2%")
    else:
        alpha = model.intercept_[0]

    results = {"alpha": alpha,
               "beta": beta,
               "x": period_ret_BM.to_list(),
               "y": period_ret_strategy.to_list()}

    if len(kwargs) == 0:
        return(results)
    else:
        return(results.get(kwargs.get("param")))

#---------------------------------------------------------------------------
                        #Maximum Drawdown
#---------------------------------------------------------------------------

def maxdd(price, pct = True):
    #Value at time T divided by max value from time t = 0 to t = T (T> 0)
    diffmax = price/ price.cummax()
    maxdd = (diffmax -1).min()

    if pct:
        return format(maxdd, ".2%")
    else:
        return maxdd

#---------------------------------------------------------------------------
                        #N Day unfiltered 99% VAR
#---------------------------------------------------------------------------

#N as number of days (integer)
def NDAYVar(price, N, pct = True):
    daily_ret = price.pct_change().fillna(0) + 1
    daily_ret = daily_ret.reset_index(drop = True)
    nday_ret = daily_ret.groupby(daily_ret.index // N).prod() - 1
    VAR = nday_ret.quantile(0.01)

    if pct:
        return format(np.abs(VAR), ".2%")
    else:
        return np.abs(VAR)

#---------------------------------------------------------------------------
                #Expected N day return or total Return
#---------------------------------------------------------------------------

def nday_ret(price, N = None, TR = False, pct = True):

    if TR:
        ret = price.tail(1)[0] / price.head(1)[0] -1
    else:
        ret = price.pct_change().mean() * N
    if pct:
        return format(ret, ".2%")
    else:
        return ret

#define return series for which ratios should be calculated:
BM_in = in_sample["benchmark_gross"]
BM_out = out_sample["benchmark_gross"]
strategy_in = in_sample["strategy_gross"]
strategy_out = out_sample["strategy_gross"]

#assemble dicts with risk return metrics
ratios_BM_in = {"Avg. Yearly Return": nday_ret(BM_in, N = 250),
         "Avg. Yearly Sharp Ratio": sharp_ratio(BM_in, days = 250, rf_rate = 0),
         "Max. Drawdown": maxdd(BM_in),
         "Alpha (monthly Returns)": "0%",
         "Beta (monthly Returns)": 1,
         "Avg. Ann. Vol": yearly_vol(BM_in, days = 250),
         "5d 99% VAR": NDAYVar(BM_in, N = 5)}

ratios_BM_out = {"Return YTD": nday_ret(BM_out, TR = True),
         "Avg. Yearly Sharp Ratio": sharp_ratio(BM_out, days = 250, rf_rate = 0),
         "Max. Drawdown": maxdd(BM_out),
         "Alpha (weekly Returns)": "0%",
         "Beta (weekly Returns)": 1,
         "Avg. Ann. Vol": yearly_vol(BM_out, days = 250),
         "5d 99% VAR": NDAYVar(BM_out, N = 5)}

ratios_strategy_in = {"Avg. Yearly Return": nday_ret(strategy_in, N = 250),
               "Avg. Yearly Sharp Ratio": sharp_ratio(strategy_in, days = 250, rf_rate = 0),
               "Max. Drawdown": maxdd(strategy_in),
               "Alpha (monthly Returns)": alpha_beta(strategy_in, BM_in, "1M", rf_period = 0, param = "alpha"),
               "Beta (monthly Returns)": round(alpha_beta(strategy_in, BM_in, "1M", rf_period = 0, param = "beta"),2),
               "Avg. Ann. Vol": yearly_vol(strategy_in, days = 250),
               "5d 99% VAR": NDAYVar(strategy_in, N = 5)}

ratios_strategy_out = {"Return YTD": nday_ret(strategy_out, TR = True),
                "Avg. Yearly Sharp Ratio": sharp_ratio(strategy_out, days = 250, rf_rate = 0),
                "Max. Drawdown": maxdd(strategy_out),
                "Alpha (weekly Returns)": alpha_beta(strategy_out, BM_out, "1W", rf_period = 0, param = "alpha"),
                "Beta (weekly Returns)": round(alpha_beta(strategy_out, BM_out, "1W", rf_period = 0, param = "beta"),2),
                "Avg. Ann. Vol": yearly_vol(strategy_out, days = 250),
                "5d 99% VAR": NDAYVar(strategy_out, N = 5)}

risk_factors_in = pd.DataFrame({"Benchmark": ratios_BM_in, "Strategy": ratios_strategy_in})
risk_factors_out = pd.DataFrame({"Benchmark": ratios_BM_out, "Strategy": ratios_strategy_out})

dfi.export(risk_factors_out, "plots/risk_factors_out.png")
dfi.export(risk_factors_in, "plots/risk_factors_in.png")

#---------------------------------------------------------------------------
        #Plot Visual Example for in Sample Alpha/ Beta Calculation
#---------------------------------------------------------------------------

#plot regression line
params = alpha_beta(strategy_in, BM_in, rf_period = 0, period = "1M", pct = False)
x = np.arange(min(params.get("x"))-0.1, max(params.get("x"))+0.1, 0.01)
fitted_y = params.get("alpha") + params.get("beta") * x

#assemble plot
fig, ax = plt.subplots(figsize = (15,10))
plt.scatter(params.get("x"), params.get("y"))
plt.plot(x, fitted_y)
plt.plot(x, x)
plt.title("Monthly Excess Returns Strategy vs. Benchmark (in sample)", size = 25)
ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
plt.ylabel("Monthly Excess Returns Strategy", size = 15)
plt.xlabel("Monthly Excess Returns Benchmark", size = 15)
plt.axhline(0, color = "black", ls = "--", lw = 1)
plt.axvline(0, color = "black", ls =  "--", lw = 1)
plt.xticks(size = 15)
plt.yticks(size = 15)
plt.legend(["liner fit", "x = y"], prop = {"size": 15})
plt.savefig("plots/in_sample_alphabetaplot.png")


#############################################################################
#                                                                           #
#                            #SECTION 4                                     #
#        #Calculate and plot portfolio characteristics vs. Benchmark        #
#                                                                           #
#############################################################################

#---------------------------------------------------------------------------
            #Excess Return Strategy vs. Benchmark Plot
#---------------------------------------------------------------------------

#calculate return over out of sample period for all stocks in the benchmark index
out_sample_all = Gross_Price[Gross_Price.index > end_backtesting]
out_sample_all = out_sample_all.pct_change() + 1
out_sample_all = out_sample_all.fillna(1).cumprod()
out_return_all = out_sample_all.iloc[len(out_sample_all.index)-1].div(out_sample_all.iloc[0]) - 1

#calculate excess return contribution as (Weight Strategy - Weigh Index) * Return
ret_contrib = stocks_index.join(stocks_invest["weights"]).fillna(0)
ret_contrib = ret_contrib[["Name", "weights", "index_weights"]]
ret_contrib["diff_weights"] = ret_contrib["weights"] - ret_contrib["index_weights"]
ret_contrib = ret_contrib.join(pd.DataFrame(out_return_all))
ret_contrib = ret_contrib.rename(columns = {0: "return"})
ret_contrib["diff_return"] = ret_contrib["diff_weights"] * ret_contrib["return"]

#plot and save results
fig = plt.figure(figsize=(15, 10))
ax = fig.add_subplot(1,1,1)
plt.bar(ret_contrib.Name[0:10], ret_contrib.diff_return[0:10])
plt.bar(ret_contrib.Name[10:39], ret_contrib.diff_return[10:39])
plt.bar(ret_contrib.Name[39:49], ret_contrib.diff_return[39:49])
plt.legend(["Long", "Not Invested", "Short", "Currencies"])
ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
plt.title("Decomposition of out of Sample excess Returns", size = 22)
plt.ylabel("Excess Return Strategy vs. Benchmark", size = 15)
plt.subplots_adjust(bottom = 0.3)
plt.xticks(rotation=90)
plt.yticks(size = 12)
plt.savefig("plots/excess_return_breakdown.png")

#---------------------------------------------------------------------------
        #Out of Sample Gross Strategy vs Benchmark Performance
#---------------------------------------------------------------------------

fig = plt.figure(figsize=(15, 10))
plt.plot(out_sample[["strategy_gross", "benchmark_gross"]])
plt.legend(out_sample[["strategy_gross", "benchmark_gross"]].columns,
           prop = {"size": 15})
plt.title("Out of Sample Strategy Performance (TR)", size = 20)
plt.ylabel("Cumulative Return", size = 15)
plt.yticks(size = 15)
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b-%Y'))
plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=1))
plt.xticks(size = 15)
plt.savefig("plots/outofsample_performance.png")

#---------------------------------------------------------------------------
        #In Sample Gross Strategy vs Benchmark Performance
#---------------------------------------------------------------------------

fig = plt.figure(figsize=(15, 10))
plt.plot(in_sample[["strategy_gross", "benchmark_gross"]])
plt.legend(in_sample[["strategy_gross", "benchmark_gross"]].columns,
           prop = {"size": 15})
plt.title("In Sample Strategy Performance", size = 22)
plt.ylabel("Cumulative Return", size = 15)
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b-%Y'))
plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3))
plt.xticks(size = 13)
plt.yticks(size = 13)
plt.savefig("plots/insample_performance.png")

#---------------------------------------------------------------------------
                  #In Sample Performance Net vs. Gross
#---------------------------------------------------------------------------

fig = plt.figure(figsize=(15, 10))
plt.plot(in_sample[["strategy_gross", "strategy_net", "benchmark_gross", "benchmark_net"]])
plt.legend(in_sample[["strategy_gross", "strategy_net", "benchmark_gross", "benchmark_net"]].columns,
           prop = {"size": 15})
plt.title("In Sample Strategy Performance (net and gross) Comparison", size = 22)
plt.ylabel("Cumulative Return", size = 15)
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b-%Y'))
plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3))
plt.xticks(size = 13)
plt.yticks(size = 13)
plt.savefig("plots/insample_performance_netgross.png")

#---------------------------------------------------------------------------
     #Out of Sample Performance Decomposition Long/ Short Portfolio
#---------------------------------------------------------------------------

fig = plt.figure(figsize=(15, 10))
plt.plot(out_sample[["strategy_gross", "benchmark_gross", "long_gross", "short_gross"]])
plt.legend(out_sample[["strategy_gross", "benchmark_gross", "long_gross", "short_gross"]].columns,
           prop = {"size": 15})
plt.title("Decomposition of out of sample performance", size = 22)
plt.ylabel("Cumulative Return", size = 15)
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b-%Y'))
plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=1))
plt.xticks(size = 13)
plt.yticks(size = 13)
plt.savefig("plots/outofsamlpe_brekdown.png")

#---------------------------------------------------------------------------
    #Stock Weights Index vs. Constrained & Unconstrained Optimization
#---------------------------------------------------------------------------

width = 0.3
fig, ax = plt.subplots(figsize = (15,10))
ind = np.arange(len(stocks_invest))
ax.barh(ind, stocks_invest.weights, width, label = "Strategy")
ax.barh(ind + width, stocks_invest.index_weights, width, label = "Index")
ax.barh(ind + 2* width, stocks_invest.weights_unconst, width, label = "Unconstrained Strategy")
ax.set(yticks = ind + width, yticklabels = stocks_invest.Name)
ax.legend(prop={'size': 10})
ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
plt.title("Stock Weights Strategy vs Index", size = 25)
plt.yticks(size = 10)
plt.xticks(size = 15, ticks = np.arange(-0.4, 0.6, 0.1))
plt.savefig("plots/strategy_weights.png")

#---------------------------------------------------------------------------
             #Plot Country Weights Index vs. Strategy
#---------------------------------------------------------------------------

country_weights_strategy = stocks_invest[["Country", "weights"]].groupby(by = "Country").sum()
country_weights_index = stocks_index[["Country", "index_weights"]].groupby(by = "Country").sum()
country_weights = country_weights_index.join(country_weights_strategy).fillna(0)
country_weights = pd.DataFrame(country_weights, index = country_weights_index.index)

#assemble plot
width = 0.3
fig, ax = plt.subplots(figsize = (15,10))
ind = np.arange(len(country_weights))
ax.barh(ind, country_weights.weights, width, label = "Strategy")
ax.barh(ind + width, country_weights.index_weights, width, label = "Index")
ax.set(yticks = ind + width, yticklabels = country_weights.index)
ax.legend(prop={'size': 15})
ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
plt.title("Country Weights Strategy vs Index", size = 25)
plt.yticks(size = 15)
plt.xticks(size = 15)
plt.savefig("plots/country_weights.png")

#---------------------------------------------------------------------------
#Plot Currency weights index vs. Strategy
#---------------------------------------------------------------------------

values = stocks_index.join(stocks_invest["weights"]).fillna(0)
currency_weights_strategy = values[["Currency", "weights"]].groupby(by = "Currency").sum()
currency_weights_index = values[["Currency", "index_weights"]].groupby(by = "Currency").sum()

currency_weights = currency_weights_index.join(currency_weights_strategy).fillna(0)
currency_weights = currency_weights.rename(index = {"GBp": "GBP"})

#assemble plot
width = 0.4
fig, ax = plt.subplots()
ind = np.arange(len(currency_weights_index))
ax.barh(ind, currency_weights.weights, width, label = "Strategy")
ax.barh(ind + width, currency_weights.index_weights, width, label = "Index")
ax.set(yticks = ind + width, yticklabels = currency_weights.index)
ax.legend(["Strategy", "Index"])
plt.title("Currency Weights Strategy vs Index", size = 20)
ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
plt.yticks(size = 15)
plt.xticks(size = 15)
plt.savefig("plots/curency_comparison.png")

#---------------------------------------------------------------------------
                #Plot sector weights strategy vs. index
#---------------------------------------------------------------------------

sector_weights_strategy = stocks_invest[["Sector", "weights"]].groupby(by = "Sector").sum()
sector_weights_index = stocks_index[["Sector", "index_weights"]].groupby(by = "Sector").sum()
sector_weights = sector_weights_index.join(sector_weights_strategy)

#assemble plot
width = 0.4
fig, ax = plt.subplots(figsize=(15, 10))
ind = np.arange(len(sector_weights))
ax.barh(ind + width, sector_weights.weights, width, label = "Strategy")
ax.barh(ind, sector_weights.index_weights, width, label = "Index")
ax.set(yticks = ind + width, yticklabels = sector_weights.index)
ax.legend(prop = {"size": 20})
plt.title("Sector Weights Strategy vs Index", size = 25)
ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
plt.yticks(size = 15)
plt.xticks(size = 15)
plt.subplots_adjust(left = 0.2)
plt.savefig("plots/sector_weights.png")

#---------------------------------------------------------------------------
    #Plot performance Nasdaq100 vs. MSCI World vs. FTSE AW-HighDivYield
#---------------------------------------------------------------------------

#download prices and clean data
prices = yf.download(tickers = ["VHYL.AS", "EXXT.DE", "IWDA.AS"], start = "2022-01-01", end = end_out_sample)
prices = prices["Adj Close"]
prices = prices.div(prices.iloc[0])
prices = prices.rename(columns = {"EXXT.DE": "Nasdaq 100", "IWDA.AS": "MSCI World", "VHYL.AS": "FTSE All World High Dividend"})

#assemble plot
plt.figure(figsize=(15, 10))
plt.plot(prices)
plt.title("High Dividend Stocks vs. Technology Stocks vs. Total Market", size = 25)
plt.legend(prices.columns,  prop={'size': 15})
plt.ylabel("Cumulative Return", size = 15)
plt.xticks(size = 15)
plt.yticks(size = 15)
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b-%Y'))
plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=1))
plt.savefig("plots/comparison.png")

#---------------------------------------------------------------------------
    #calculate change of correlation  matrix in vs out of sample
#---------------------------------------------------------------------------

out_sample_cor = Gross_Price_selected[Gross_Price_selected.index > end_backtesting].pct_change().corr()
in_sample_cor = Gross_Price_selected[Gross_Price_selected.index < end_backtesting].pct_change().corr()

rel_cor = out_sample_cor / in_sample_cor
pd.DataFrame(rel_cor).to_csv("files/correlation_change.csv")

#---------------------------------------------------------------------------
  #calculate performance ratios for short and long portfolios & Benchmark
#---------------------------------------------------------------------------

"""
we only include stocks which contain a forward,trailing PE ratio &
PB Ratio to calculate weighted values. 
Further, as described by iShares,
we restrict the maximum value of PE ratios to 60 and of PB ratios to 25. 
"""
stocks_short = stocks_invest[stocks_invest.weights <0]
stocks_long = stocks_invest[stocks_invest.weights > 0]

#dividend yield
div_yield_long = (stocks_long.weights / stocks_long.weights.sum() * stocks_long.Yield).sum()
div_yield_short = (stocks_short.weights.abs()/ stocks_short.weights.abs().sum() * stocks_short.Yield).sum()
div_yield_index = (stocks_index.index_weights / stocks_index.index_weights.sum() * stocks_index.Yield).sum()

#trailing PE Ratio, forward PE, PB Ratio

#get all stocks and join strategy weights
index = stocks_index.join(stocks_invest[["weights"]])

#1. remove stocks where ratios are not given
index = index[index.Trailing_PE.notna()]
index = index[index.Forward_PE.notna()]

#2. set PE ratios above 60 to 60, set PB ratios above 25 to 25
index.Trailing_PE = np.where(index.Trailing_PE > 60, 60, index.Trailing_PE)
index.Forward_PE = np.where(index.Forward_PE > 60, 60, index.Forward_PE)
index.PB_Ratio = np.where(index.PB_Ratio > 25, 25, index.PB_Ratio)

#3. separate long and short stocks from strategy to calculate ratios separately
short = index.loc[stocks_short.index]
long = index.loc[stocks_long.index]

#calculate weighted Trailing PE
PE_long = (long.weights / long.weights.sum() * long.Trailing_PE).sum()
PE_short = (short.weights / short.weights.sum() * short.Trailing_PE).sum()
PE_index = (index.index_weights / index.index_weights.sum() * index.Trailing_PE).sum()

#calculate weighted forward PE
PE_fwd_long = (long.weights / long.weights.sum() * long.Forward_PE).sum()
PE_fwd_short = (short.weights / short.weights.sum() * short.Forward_PE).sum()
PE_fwd_index = (index.index_weights / index.index_weights.sum() * index.Forward_PE).sum()

#calculate PB Ratio
PB_long = (long.weights / long.weights.sum() * long.PB_Ratio).sum()
PB_short = (short.weights / short.weights.sum() * short.PB_Ratio).sum()
PB_index = (index.index_weights / index.index_weights.sum() * index.PB_Ratio).sum()

#assemble metrics dataframe
ratios_short = {"Yield": format(div_yield_short,".2%"),
                "Price_Book": round(PB_short,2),
                "Trailing_PE": round(PE_short,2),
                "Forward_PE": round(PE_fwd_short,2)}

ratios_long = {"Yield": format(div_yield_long,".2%"),
               "Price_Book": round(PB_long,2),
               "Trailing_PE": round(PE_long,2),
               "Forward_PE": round(PE_fwd_long,2)}

ratios_index = {"Yield": format(div_yield_index,".2%"),
                "Price_Book": round(PB_index,2),
                "Trailing_PE": round(PE_index,2),
                "Forward_PE": round(PE_fwd_index,2)}

ratios_table = pd.DataFrame({"Portfolio Short":ratios_short, "Portfolio Long": ratios_long, "Index": ratios_index})

#export table
dfi.export(ratios_table, "plots/portfolio_characteristics.png")
