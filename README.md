## Description 

This script was written for a group project for the Class Investments at NCCU. 
Team Members: 

-
-
-
-

## Motivation

We implement a quantitative equity strategy whereby we select stocks from the Stoxx Europe 50 index which comprises of some of the largest companies in Europe.

Inspired by the "dogs of the dow strategy which predicts an outperformance of the 10 highest dividend yielding stocks in the dow vs the total index, we restrict the portfolio to be long in the top 10 dividend yielding stocks in the stoxx europe 50. Additionally, we go short in the 10 lowest yielding dividend stocks. This is motivated by the believe that after our in sample period ends (31.12.2021), rising interst rates and inflation will lead to a correlation breakdown between low and high dividend yield stocks where high dividend yield stocks will outperform short dividend yield stocks.

## Implementation

The quantitative optimization technique used to genereate our asset allocation weights is based on Modern Oprtfolio Theory whereby we try to find the Maximum Sharp Ratio Portfolio. 

In the Optimnization, we include the following restrictions on our portfolio: 

- all long (short) investments  carry a minimum weight of 2% & (-2%)
- The maximum value for long (short) investmnets is 20% (-20%)
- The total allocation in stocks is 100% (no leverage, 100% invested)

our backtesting period is: 2019 - end of 2021 and our out of sample period is: start of 2022 - 24.05.2022

## Results 
The table shows the selected stocks in the strategy, some quantitative ratios such as their yields or PE ratios as well as the weights allocated in our strategy and the weights that would have been used had we not implemented resrictions beyond the 100% total allocation restriction.

<center>
  <img src="plots/selected_portfolio_characteristics.png" alt="drawing" width="800"/>
</center>

Our strategy significantly outperforms the benchmark in the in-sample as well as out-sample period. Further, most of the risk - return metrics are superior for the strategy as compared to the benchmark.

<p float="left">
  <img src="plots/insample_performance.png" width="400" />
  <img src="plots/outofsample_performance.png" width="400" /> 
</p>

<p float="left">
    &nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp; 
<img src="plots/risk_factors_in.png" width="300" />
  &nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;
  <img src="plots/risk_factors_out.png" width="300" /> 
</p>
