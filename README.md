# Investments_Project

## Motivation

We implement a quantitative equity strategy whereby we select stocks from the Stoxx Europe 50 index which comprises of some of the largest companies in Europe.

Inspired by the "dogs of the dow strategy which predicts an outperformance of the 10 highest dividend yielding stocks in the dow vs the total index, we restrict the portfolio to be long in the top 10 dividend yielding stocks in the stoxx europe 50. Additionally, we go short in the 10 lowest yielding dividend stocks. 

<img src="plots/mystocks.png" alt="drawing" width="800"/>

The motivation behind shorting the 10 lowest paying dividend stocks is the believe that in 2022, inflation and thus interest rates will rise significantly, this ....

## Implementation

We caltulate the optimal weights by calculating the Maximum Sharp Ratio Portfolio with the following restrictions: 

- all long (short) investments  carry a minimum weight of 2& (-2%)
- The maximum value for long (short) investmnets is 20% (-20%)

- our backtesting period is: XXXX
- our out of sample period is: start of 2022 - 24.05.2022

## Results 
Results: 

<img src="mystocks.png" alt="drawing" width="800"/>



