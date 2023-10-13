# Time Series Analysis



##### Chapter 1
##### Section 1
##### Intro to the course



Introduction to Time Series


#Pandas tools to help

# Changing Index of a dataframe to Datetime
df.index = pd.to_datetime(df.index)

# Quick way to plot data Plotting Data
df.plot()

# You can sclice the data by year and then plot it
df['2012']

# join two data frames
df1.join(df2)

# resample data (eg from daily to weekly)
df = df.resample(rule='W').last()

# computing percent changes and differences of a time series
df['col'].pct_change()
df['col'].diff()

# compute the correlation of a series using the correlation method
df['ABC'].corr(df['XYV'])

# autocorrelcation
df['ABC'].autocorr()


##### Chapter 1
##### Section 1
##### Exercises

# Import pandas and plotting modules
import pandas as pd
import matplotlib.pyplot as plt
# Convert the date index to datetime
diet.index = pd.to_datetime(diet.index)
# Plot the entire time series diet and show gridlines
diet.plot(grid=True)
plt.show()
# Slice the dataset to keep only 2012
diet2012 = diet['2012']
# Plot 2012 data
diet2012.plot(grid=True)
plt.show()

# Import pandas
import pandas as pd
# Convert the stock index and bond index into sets
set_stock_dates = set(stocks.index)
set_bond_dates = set(bonds.index)
# Take the difference between the sets and print
print(set_stock_dates - set_bond_dates)
# Merge stocks and bonds DataFrames using join()
stocks_and_bonds = stocks.join(bonds, how='inner')



##### Chapter 1
##### Section 2
##### Correlation of Two Time Series

#Correlation of Large Cap and Small Cap Stocks

# First: Computer percentage change on both series
df['SPX_Ret'] = df['SPX_Prices'].pct_change()
df['R2000_Ret'] = df['R2000_Prices'].pct_change()

#Visualiza correlation with scatter plot
plt.scatter(df['SPX_Ret'], df['R2000_Ret'])
plt.show()

# Use pandas correlation method for Series
correlation = df['SPX_Ret'].corr(df['R2000_Ret'])
print("Correlation is: ", correlation)


##### Chapter 1
##### Section 2
##### Exercises

# Compute percent change using pct_change()
returns = stocks_and_bonds.pct_change()
# Compute correlation using corr()
correlation = returns['SP500'].corr(returns['US10Y'])
print("Correlation of stocks and interest rates: ", correlation)
# Make scatter plot
plt.scatter(returns['SP500'], returns['US10Y'])
plt.show()

# Compute correlation of levels
correlation1 = levels['DJI'].corr(levels['UFO'])
print("Correlation of levels: ", correlation1)
# Compute correlation of percent changes
changes = levels.pct_change()
correlation2 = changes['DJI'].corr(changes['UFO'])
print("Correlation of changes: ", correlation2)



##### Chapter 1
##### Section 3
##### Simple Linear Regression

# Python Packages to Perform Regressions
# In statsmodels:
import statsmodels.api as sm
sm.OLS(y,x).fit()
# In numpy
np.polyfit(x,y, deg=1)
# In pandas
pd.ols(y, x)
# In scipy
from scipy import stats
stats.linregress(x,y)

# EXAMPLE
# Import the statsmodels module
import statsmodels.api as sm
# As before, compute percentage changes in both series
df['SPX_Ret'] = df['SPX_Prices'].pct_change()
df['R2000_Ret'] = df['R2000_Prices'].pct_change()
# Add a constant to the DataFramge for the regression intercept
# This has to be done or the stats model will assume we're doing this without an intercept
df = sm.add_constant(df)

# Pandas will return a blank first row, so we drop it
df = df.dropna()
# Run the Regression
results = sm.OLS(df['R2000_Ret'], df[['const', 'SPX_Ret']]).fit()
print(results.summary())
# intercept
results.params[0]
# slope
results.params[1]


##### Chapter 1
##### Section 3
##### Exercise

# Import the statsmodels module
import statsmodels.api as sm
# Compute correlation of x and y
correlation = x.corr(y)
print("The correlation between x and y is %4.2f" %(correlation))
# Convert the Series x to a DataFrame and name the column x
dfx = pd.DataFrame(x, columns=['x'])
# Add a constant to the DataFrame dfx
dfx1 = sm.add_constant(dfx)
# Regress y on dfx1
result = sm.OLS(y, dfx1).fit()
# Print out the results and look at the relationship between R-squared and the correlation above
print(result.summary())



##### Chapter 1
##### Section 4
##### Autocorrelation

"""
Autocorrelation is a correlation of a time series with a lagged copy of itself
Also called Serial Correlation

Mean Reversion - Negative Autocorrelation
Momentum or Trend Following - Positive Autocorrelation

"""
# EXAMPLE of Positive Autocorrelation: Exchange Rates
# Convert index to datetime
df.index = pd.to_datetime(df.index)
# Downsample from daily to monthly data
df = df.resample(rules='M').last() # could use first, last, or average
# Computer returns from prices
df['Return'] = df['Price'].pct_exchange()
# Compute autocorrelation
autocorrelation = df['Return'].autocorr()
print("The autocorrelation is: ", autocorrelation)


##### Chapter 1
##### Section 4
##### Exercises

# Convert the daily data to weekly data
MSFT = MSFT.resample(rule='W').last()
# Compute the percentage change of prices
returns = MSFT.pct_change()
# Compute and print the autocorrelation of returns
autocorrelation = returns['Adj Close'].autocorr()
print("The autocorrelation of weekly returns is %4.2f" %(autocorrelation))

# Compute the daily change in interest rates 
daily_diff = daily_rates.diff()
# Compute and print the autocorrelation of daily changes
autocorrelation_daily = daily_diff['US10Y'].autocorr()
print("The autocorrelation of daily interest rate changes is %4.2f" %(autocorrelation_daily))
# Convert the daily data to annual data
yearly_rates = daily_rates.resample(rule='A').last()
# Repeat above for annual data
yearly_diff = yearly_rates.diff()
autocorrelation_yearly = yearly_diff['US10Y'].autocorr()
print("The autocorrelation of annual interest rate changes is %4.2f" %(autocorrelation_yearly))




##### Chapter 2
##### Section 1
##### Autocorrelation Function

# TO PLOT THE AUTO CORRELATION FUNCTION
#import the module
from statsmodels.graphics.tsaplots import plot_acf
#plot the ACF # lags is number of lags, alpha is a confidence interval
plot_acf(x, lags=20, alpha=0.05) 

# pl

# To see ACF values instead of plot
from statsmodels.tsa.stattools import acf
print(acf(x))


##### Chapter 2
##### Section 1
##### Exercises

# Import the acf module and the plot_acf module from statsmodels
from statsmodels.tsa.stattools import acf
from statsmodels.graphics.tsaplots import plot_acf
# Compute the acf array of HRB
acf_array = acf(HRB)
print(acf_array)
# Plot the acf function
plot_acf(HRB, alpha=1)
plt.show()

# Import the plot_acf module from statsmodels and sqrt from math
from statsmodels.graphics.tsaplots import plot_acf
from math import sqrt
# Compute and print the autocorrelation of MSFT weekly returns
autocorrelation = returns['Adj Close'].autocorr()
print("The autocorrelation of weekly MSFT returns is %4.2f" %(autocorrelation))
# Find the number of observations by taking the length of the returns DataFrame
nobs = len(returns)
# Compute the approximate confidence interval
conf = 1.96/sqrt(nobs)
print("The approximate confidence interval is +/- %4.2f" %(conf))
# Plot the autocorrelation function with 95% confidence intervals and 20 lags using plot_acf
plot_acf(returns, lags=20, alpha=0.05)
plt.show()


##### Chapter 2
##### Section 2
##### White Noise

# Generate White noise
import numpy as np
noise = np.random.normal(loc=0, scale=1, size=500)
# loc is the mean
# scale is the standard deviation

# The correlation of all white noise is zero

# plot the white noise
plt.plot(noise)

plot_acf(noise, lags=50)


##### Chapter 2
##### Section 2
##### Exercises

# Import the plot_acf module from statsmodels
from statsmodels.graphics.tsaplots import plot_acf
# Simulate white noise returns
returns = np.random.normal(loc=0.02, scale=0.05, size=1000)
# Print out the mean and standard deviation of returns
mean = np.mean(returns)
std = np.std(returns)
print("The mean is %5.3f and the standard deviation is %5.3f" %(mean,std))
# Plot returns series
plt.plot(returns)
plt.show()
# Plot autocorrelation function of white noise returns
plot_acf(returns, lags=20)
plt.show()


##### Chapter 2
##### Section 3
##### Random Walk

"""
dickey-fuller test, determining if two things are random together.
P_t - P_t-1 = a +BP_t-1 + E

If B = 0 (It's a random Walk)
If B < 0 (Not a random walk)
"""
from statsmodels.tsa.stattools import adfuller
adfuller(x)

# Example
results = adfuller(df['SPX'])
print(results[1]) # Print p-value
print(results) # Print full results


##### Chapter 2
##### Section 3
##### Exercise

# Generate 500 random steps with mean=0 and standard deviation=1
steps = np.random.normal(loc=0, scale=1, size=500)
# Set first element to 0 so that the first price will be the starting stock price
steps[0]=0
# Simulate stock prices, P with a starting price of 100
P = 100 + np.cumsum(steps)
# Plot the simulated stock prices
plt.plot(P)
plt.title("Simulated Random Walk")
plt.show()

# Generate 500 random steps
steps = np.random.normal(loc=.001, scale=0.01, size=500) + 1
# Set first element to 1
steps[0]=1
# Simulate the stock price, P, by taking the cumulative product
P = 100 * np.cumprod(steps)
# Plot the simulated stock prices
plt.plot(P)
plt.title("Simulated Random Walk with Drift")
plt.show()

# Import the adfuller module from statsmodels
from statsmodels.tsa.stattools import adfuller
# Run the ADF test on the price series and print out the results
results = adfuller(AMZN['Adj Close'])
print(results)
# Just print out the p-value
print('The p-value of the test on prices is: ' + str(results[1]))

# Import the adfuller module from statsmodels
from statsmodels.tsa.stattools import adfuller
# Create a DataFrame of AMZN returns
AMZN_ret = AMZN.pct_change()
# Eliminate the NaN in the first row of returns
AMZN_ret = AMZN_ret.dropna()
# Run the ADF test on the return series and print out the p-value
results = adfuller(AMZN_ret)
print('The p-value of the test on returns is: ' + str(results[1]))


##### Chapter 2
##### Section 4
##### Stationarity
"""
Stationairity- The observations don't depend on time
Weak Stationarity- The mean, variance, and autocorrelations do not depend on time

If parameters vary with time, there are too many to estimate

"""

plot.plot(SPY)
plot.plot(SPY.diff())



##### Chapter 2
##### Section 4
##### Exercises

# Import the plot_acf module from statsmodels
from statsmodels.graphics.tsaplots import plot_acf
# Seasonally adjust quarterly earnings
HRBsa = HRB.diff(4)
# Print the first 10 rows of the seasonally adjusted series
print(HRBsa.head(10))
# Drop the NaN data in the first four rows
HRBsa = HRBsa.dropna()
# Plot the autocorrelation function of the seasonally adjusted series
plot_acf(HRBsa)
plt.show()


##### Chapter 3
##### Section 1
##### Describe AR Model

"""
Auto Regressive Model

AR(1) Model
If Phi is 1, it's a random walk
If Phi is 0, it's white noise
-1 < phi < 1

Positive 
Mean Reversion

"""
from statsmodels.tsa.arima_process import ArmaProcess
ar = np.array([1, -0.9])
ma = np.array([1])
AR_object = ArmaProcess(ar, ma)
simulated_data = AR_object.generate_sample(nsample=1000)
plt.plot(simulated_data)

##### Chapter 3
##### Section 1
##### Exercises

# import the module for simulating data
from statsmodels.tsa.arima_process import ArmaProcess

# Plot 1: AR parameter = +0.9
plt.subplot(2,1,1)
ar1 = np.array([1, -0.9])
ma1 = np.array([1])
AR_object1 = ArmaProcess(ar1, ma1)
simulated_data_1 = AR_object1.generate_sample(nsample=1000)
plt.plot(simulated_data_1)
# Plot 2: AR parameter = -0.9
plt.subplot(2,1,2)
ar2 = np.array([1, 0.9])
ma2 = np.array([1])
AR_object2 = ArmaProcess(ar2, ma2)
simulated_data_2 = AR_object2.generate_sample(nsample=1000)
plt.plot(simulated_data_2)
plt.show()


# Import the plot_acf module from statsmodels
from statsmodels.graphics.tsaplots import plot_acf
# Plot 1: AR parameter = +0.9
plot_acf(simulated_data_1, alpha=1, lags=20)
plt.show()
# Plot 2: AR parameter = -0.9
plot_acf(simulated_data_2, alpha=1, lags=20)
plt.show()
# Plot 3: AR parameter = +0.3
plot_acf(simulated_data_3, alpha=1, lags=20)
plt.show()


##### Chapter 3
##### Section 2
##### Estimating and forecasting an AR Model

# Estimating an AR Model
from statsmodels.tsa.arima_model import ARMA
mod = ARMA(data, order=(1,0)) # order=(p,q)
result = mod.fit()

from statsmodels.tsa.arima.model import arima
mod = ARIMA(data, order=(1,0,0)) # order = (p,d,q)
# p is degree of model
# d are we taking first differences?
# q is the MA part
result = mod.fit()

print(result.summary()) 
print(result.params) # Returns mu and phi

# Forcasting with an AR model
from statsmodels.graphics.tsaplots import plot_predict
fig, ax = plt.subplots()
data.plot(ax=ax)
plot_predict(result, start='2012-09-27', end='2012-10-06', alpha=0.05, ax=ax)
plt.show()


##### Chapter 3
##### Section 2
##### Exercises

# Import the ARIMA module from statsmodels
from statsmodels.tsa.arima.model import ARIMA
# Fit an AR(1) model to the first simulated data
mod = ARIMA(simulated_data_1, order=(1,0,0))
res = mod.fit()
# Print out summary information on the fit
print(res.summary())
# Print out the estimate for phi
print("When the true phi=0.9, the estimate of phi is:")
print(res.params[1])

# Import the ARIMA and plot_predict from statsmodels
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_predict
# Forecast the first AR(1) model
mod = ARIMA(simulated_data_1, order=(1,0,0))
res = mod.fit()
# Plot the data and the forecast
fig, ax = plt.subplots()
simulated_data_1.loc[950:].plot(ax=ax)
plot_predict(res, start=1000, end=1010, ax=ax)
plt.show()

# Forecast interst rates using an AR(1) model
mod = ARIMA(interest_rate_data, order=(1,0,0))
res = mod.fit()
# Plot the data and the forecast
fig, ax = plt.subplots()
interest_rate_data.plot(ax=ax)
plot_predict(res, start=0, end='2027', alpha=None, ax=ax)
plt.show()

# Import the plot_acf module from statsmodels
from statsmodels.graphics.tsaplots import plot_acf
# Plot the interest rate series and the simulated random walk series side-by-side
fig, axes = plt.subplots(2,1)
# Plot the autocorrelation of the interest rate series in the top plot
fig = plot_acf(interest_rate_data, alpha=1, lags=12, ax=axes[0])
# Plot the autocorrelation of the simulated random walk series in the bottom plot
fig = plot_acf(simulated_data, alpha=1, lags=12, ax=axes[1])
# Label axes
axes[0].set_title("Interest Rate Data")
axes[1].set_title("Simulated Random Walk Data")
plt.show()



##### Chapter 3
##### Section 3
##### Choosing the Right Model

# Partial Autocorrelation Function (PACF)
plot_pact
plot_pact(x, lags=20, alpha=0.05)

# For evaluating a model
from statsmodels.tsa.arima_model import ARIMA
mod = ARIMA(simulated_data, order(1,0))
result = mod.fit()
result.summary()
result.params
result.aic
result.bic 


##### Chapter 3
##### Section 3
##### Exercise

# Import the modules for simulating data and for plotting the PACF
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.graphics.tsaplots import plot_pacf
# Simulate AR(1) with phi=+0.6
ma = np.array([1])
ar = np.array([1, -0.6])
AR_object = ArmaProcess(ar, ma)
simulated_data_1 = AR_object.generate_sample(nsample=5000)
# Plot PACF for AR(1)
plot_pacf(simulated_data_1, lags=20)
plt.show()
# Simulate AR(2) with phi1=+0.6, phi2=+0.3
ma = np.array([1])
ar = np.array([1, -0.6, -0.3])
AR_object = ArmaProcess(ar, ma)
simulated_data_2 = AR_object.generate_sample(nsample=5000)
# Plot PACF for AR(2)
plot_pacf(simulated_data_2, lags=20)
plt.show()

# Import the module for estimating an ARIMA model
from statsmodels.tsa.arima.model import ARIMA
# Fit the data to an AR(p) for p = 0,...,6 , and save the BIC
BIC = np.zeros(7)
for p in range(7):
    mod = ARIMA(simulated_data_2, order=(p,0,0))
    res = mod.fit()
# Save BIC for AR(p)    
    BIC[p] = res.bic
    # Plot the BIC as a function of p
plt.plot(range(1,7), BIC[1:7], marker='o')
plt.xlabel('Order of AR Model')
plt.ylabel('Bayesian Information Criterion')
plt.show()



##### Chapter 4
##### Section 1
##### Describe Model
# Moving Average function

# simulating an MA Priocess
from statsmodels.tsa.arima.arima_process import ArmaProcess
ar = np.array([1])
ma = np.array([1, 0.5])
AR_object = ArmaProcess(ar,ma)
simulated_data = AR_object.generate_sample(nsample=1000)
plt.plot(simulated_data)


##### Chapter 4
##### Section 1
##### Exercises

# import the module for simulating data
from statsmodels.tsa.arima_process import ArmaProcess
# Plot 1: MA parameter = -0.9
plt.subplot(2,1,1)
ar1 = np.array([1])
ma1 = np.array([1, -0.9])
MA_object1 = ArmaProcess(ar1, ma1)
simulated_data_1 = MA_object1.generate_sample(nsample=1000)
plt.plot(simulated_data_1)
# Plot 2: MA parameter = +0.9
plt.subplot(2,1,2)
ar2 = np.array([1])
ma2 = np.array([1, 0.9])
MA_object2 = ArmaProcess(ar2, ma2)
simulated_data_2 = MA_object2.generate_sample(nsample=1000)
plt.plot(simulated_data_2)
plt.show()

# Import the plot_acf module from statsmodels
from statsmodels.graphics.tsaplots import plot_acf
# Plot 1: MA parameter = -0.9
plot_acf(simulated_data_1, lags=20)
plt.show()
# Plot 2: MA parameter = 0.9
plot_acf(simulated_data_2, lags=20)
plt.show()
# Plot 3: MA parameter = -0.3
plot_acf(simulated_data_3, lags=20)
plt.show()


##### Chapter 4
##### Section 2
##### Estimating an MA Model

from statsmodels.tsa.arima.model import ARIMA
mod = ARIMA(simulated_data, order=(0,0,1)) # 1 for an ar model
result = mod.fit()

#forecasting an MA model
from statsmodels.graphics.tsaplots import plot_predict
fig, ax = plt.subplots()
data.plot(ax=ax)
plot_predict(res, start='2012-09-27', end='2012-10-06', ax=ax)
plt.show()


##### Chapter 4
##### Section 2
##### Exercises

# Import the ARIMA module from statsmodels
from statsmodels.tsa.arima.model import ARIMA

# Fit an MA(1) model to the first simulated data
mod = ARIMA(simulated_data_1, order=(0,0,1))
res = mod.fit()
# Print out summary information on the fit
print(res.summary())
# Print out the estimate for the constant and for theta
print("When the true theta=-0.9, the estimate of theta is:")
print(res.params[1])


# Import the ARIMA and plot_predict from statsmodels
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_predict
# Forecast the first MA(1) model
mod = ARIMA(simulated_data_1, order=(0,0,1))
res = mod.fit()
# Plot the data and the forecast
fig, ax = plt.subplots()
simulated_data_1.loc[950:].plot(ax=ax)
plot_predict(res, start=1000, end=1010, ax=ax)
plt.show()


##### Chapter 4
##### Section 3
##### ARMA Models

# an ARMA model is a combination of AR and MA Models


##### Chapter 4
##### Section 3
##### Exercises

# import datetime module
import datetime
# Change the first date to zero
intraday.iloc[0,0] = 0
# Change the column headers to 'DATE' and 'CLOSE'
intraday.columns = ['DATE', 'CLOSE']
# Examine the data types for each column
print(intraday.dtypes)
# Convert DATE column to numeric
intraday['DATE'] = pd.to_numeric(intraday['DATE'])
# Make the `DATE` column the new index
intraday = intraday.set_index('DATE')
# Notice that some rows are missing
print("If there were no missing rows, there would be 391 rows of minute data")
print("The actual length of the DataFrame is:", len(intraday))
# Everything
set_everything = set(range(391))
# The intraday index as a set
set_intraday = set(intraday.index)
# Calculate the difference
set_missing = set_everything - set_intraday
# Print the difference
print("Missing rows: ", set_missing)
# From previous step
intraday = intraday.reindex(range(391), method='ffill')
# Change the index to the intraday times
intraday.index = pd.date_range(start='2017-09-01 9:30', end='2017-09-01 16:00', freq='1min')
# Plot the intraday time series
intraday.plot(grid=True)
plt.show()

# Import plot_acf and ARIMA modules from statsmodels
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.arima.model import ARIMA
# Compute returns from prices and drop the NaN
returns = intraday.pct_change()
returns = returns.dropna()
# Plot ACF of returns with lags up to 60 minutes
plot_acf(returns, lags=60)
plt.show()
# Fit the data to an MA(1) model
mod = ARIMA(returns, order=(0,0,1))
res = mod.fit()
print(res.params[1])


# import the modules for simulating data and plotting the ACF
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.graphics.tsaplots import plot_acf
# Build a list MA parameters
ma = [0.8 ** i for i in range(30)]
# Simulate the MA(30) model
ar = np.array([1])
AR_object = ArmaProcess(ar, ma)
simulated_data = AR_object.generate_sample(nsample=5000)
# Plot the ACF
plot_acf(simulated_data, lags=30)
plt.show()


##### Chapter 5
##### Section 1
##### Cointegration Models

from statsmodels.tsa.stattools import coint
coint(P,Q)


##### Chapter 5
##### Section 1
##### Exercises

# Plot the prices separately
plt.subplot(2,1,1)
plt.plot(7.25*HO, label='Heating Oil')
plt.plot(NG, label='Natural Gas')
plt.legend(loc='best', fontsize='small')

# Plot the spread
plt.subplot(2,1,2)
plt.plot(7.25*HO-NG, label='Spread')
plt.legend(loc='best', fontsize='small')
plt.axhline(y=0, linestyle='--', color='k')
plt.show()

# Import the adfuller module from statsmodels
from statsmodels.tsa.stattools import adfuller
# Compute the ADF for HO and NG
result_HO = adfuller(HO['Close'])
print("The p-value for the ADF test on HO is ", result_HO[1])
result_NG = adfuller(NG['Close'])
print("The p-value for the ADF test on NG is ", result_NG[1])
# Compute the ADF of the spread
result_spread = adfuller(7.25 * HO - NG)
print("The p-value for the ADF test on the spread is ", result_spread[1])

# Import the statsmodels module for regression and the adfuller function
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
# Regress BTC on ETH
ETH = sm.add_constant(ETH)
result = sm.OLS(BTC,ETH).fit()
# Compute ADF
b = result.params[1]
adf_stats = adfuller(BTC['Price'] - b*ETH['Price'])
print("The p-value for the ADF test is ", adf_stats[1])

##### Chapter 5
##### Section 2
##### Climate Change Case Study Exercises

# Import the adfuller function from the statsmodels module
from statsmodels.tsa.stattools import adfuller

# Convert the index to a datetime object
temp_NY.index = pd.to_datetime(temp_NY.index, format='%Y')

# Plot average temperatures
temp_NY.plot()
plt.show()

# Compute and print ADF p-value
result = adfuller(temp_NY['TAVG'])
print("The p-value for the ADF test is ", result[1])



# Import the modules for plotting the sample ACF and PACF
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
# Take first difference of the temperature Series
chg_temp = temp_NY.diff()
chg_temp = chg_temp.dropna()
# Plot the ACF and PACF on the same page
fig, axes = plt.subplots(2,1)
# Plot the ACF
plot_acf(chg_temp, lags=20, ax=axes[0])
# Plot the PACF
plot_pacf(chg_temp, lags=20, ax=axes[1])
plt.show()


#LOWER AIC scores are better
# Import the module for estimating an ARIMA model
from statsmodels.tsa.arima.model import ARIMA

# Fit the data to an AR(1) model and print AIC:
mod_ar1 = ARIMA(chg_temp, order=(1, 0, 0))
res_ar1 = mod_ar1.fit()
print("The AIC for an AR(1) is: ", res_ar1.aic)
# Fit the data to an AR(2) model and print AIC:
mod_ar2 = ARIMA(chg_temp, order=(2, 0, 0))
res_ar2 = mod_ar2.fit()
print("The AIC for an AR(2) is: ", res_ar2.aic)
# Fit the data to an ARMA(1,1) model and print AIC:
mod_arma11 = ARIMA(chg_temp, order =(1,0,1))
res_arma11 = mod_arma11.fit()
print("The AIC for an ARMA(1,1) is: ", res_arma11.aic)


# Import the ARIMA module from statsmodels
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_predict
# Forecast temperatures using an ARIMA(1,1,1) model
mod = ARIMA(temp_NY, trend='t', order=(1,1,1))
res = mod.fit()
# Plot the original series and the forecasted series
fig, ax = plt.subplots()
temp_NY.plot(ax=ax)
plot_predict(res, start='1872', end='2046', ax=ax)
plt.show()


#####
#####
#####
##### Intro to ACF and PACF
#####
#####

##### Chapter 3 
##### Section 1
##### Intro to ACF and PACF

# ACF- Auto-correlation function 
# PACF- Partial Auto Correlation Function 
# AR- Auto-Regression Model
# MA- Moving Average Model
# ARMA- Auto Regressive-moving average model
# ARIMA- Auto Regressive Integrated Moving Average Model 

#              AR(p)                    MA(q)                       ARMA(p,q)
# ACF          Tails Off                Cutts off after lag q       Tails off
# PACF         Cuts off after lag p     Tails off                   Tails off

from statsmodel.graphics.tsaplots import plot_acf, plot_pacf
# create figure
fig, (ax1, ax2) = plt.subplots(2,1 figsize=(8,8))
# Make ACF plot
plot_acf(df, lags=10, zero = False, ax=ax1)
# Make PACF ploT
plot_pacf(df, lags=10, zero= False, ax=ax2)

plt.show()

# The time series must be made stationary before making the ACF and PACF plots.

# If the autocorrrelation is just 1 and the partial auto-correlation drops off after 1, 
# then the data is non-stationary and needs to be differenced.

# If the Autocorrelation function is negative one and the partial autocorrelation function is just negative,
# this is a sign that we've taken the difference too many times.

##### Chapter 3
##### Section 1
##### Exercises

# Create figure
fig, (ax1, ax2) = plt.subplots(2,1, figsize=(12,8))
# Plot ACF and PACF
plot_acf(earthquake,lags=15, zero=False, ax=ax1)
plot_pacf(earthquake, lags=15, zero=False, ax=ax2)
# Show plot
plt.show()


# Create figure
fig, (ax1, ax2) = plt.subplots(2,1, figsize=(12,8))
# Plot ACF and PACF
plot_acf(earthquake, lags=10, zero=False, ax=ax1)
plot_pacf(earthquake, lags=10, zero=False, ax=ax2)
# Show plot
plt.show()
# Instantiate model
model = ARIMA(earthquake, order = (1,0,0))
# Train model
results = model.fit()


##### Chapter 3
##### Section 2
##### AIC and BIC

# AIC - Akaike Information Criterion
# Lower AIC => better model
# AIC chooses simple models with lower order
# Penalizes models with a lot of parameters

# BIC - Batesian information criterion
# Lower BIC => better model
# BIC favors simpler models

# BIC favors simpler models than AIC
# AIC better for predictive models
# BIC is better for explanatory model

# create model
model = ARIMA(df, order=(1,0,1))
# Fit model 
results = model.fit()
# Print fit summary
print(results.summary())

#could also print them out
print('AIC:', results.aic)
print('BIC:', results.bic)


#searching over AIC and BIC

order_aic_bic =[]
#loop over AR order
for p in range(3):
    for q in range(3):
        try:
            model = ARIMA(df,order=(p,0,q))
            results = model.fit()
            print(p,q,results.aic, results.bic)
            order_aic_bic.append((p,q,results.aic, results.bic))
        except:
            print(p,q,None,None)

# Make is a data frame
order_df = pd.DataFrame(order_aic_bic, columns = ['p'.'q','aic', 'bic'])

#Sort by AIC
print(order_df.sort_values('aic'))

# Sort by BIC
print(order_df.sort_values('bic'))

# Fit model
model = ARIMA(df, order = (2,0,1))
results = model.fit()

# If the model results in an error
# Non-stationary


##### Chapter 3
##### Section 2
##### Exercises
# Create empty list to store search results
order_aic_bic=[]

# Loop over p values from 0-2
for p in range(3):
  # Loop over q values from 0-2
    for q in range(3):
        # create and fit ARMA(p,q) model
        model = ARIMA(df, order=(p,0,q))
        results = model.fit()
        
        # Append order and results tuple
        order_aic_bic.append((p,q, results.aic, results.bic))


# Construct DataFrame from order_aic_bic
order_df = pd.DataFrame(order_aic_bic, 
                        columns=['p','q','AIC','BIC'])

# Print order_df in order of increasing AIC
print(order_df.sort_values('AIC'))
# Print order_df in order of increasing BIC
print(order_df.sort_values('BIC'))

# Loop over p values from 0-2
for p in range(3):
    # Loop over q values from 0-2
    for q in range(3):
      
        try:
            # create and fit ARMA(p,q) model
            model = ARIMA(earthquake, order=(p,0,q))
            results = model.fit()
            
            # Print order and results
            print(p, q, results.aic, results.bic)
            
        except:
            print(p, q, None, None)     


##### Chapter 3
##### Section 3
##### Model Diagnostics

# How good is the model?
# Residuals - our models one step ahead predictions and the real values of the time series. 

model = ARIMA(df, order = (p,d,q))
results = model.fit()
residuals = results.resid

# Mean absolute Error of the residuals
mae = np.mean(np.abs(residuals))

# If the model fits well, the residuals will be white noise

results.plot_diagnostics()
plt.show()

1. Standardized residuals - should have no obvious pattern or structure
2. Distribution of the residuals - If our model is good, both lines should be really close to each plot 
3. Normal Q-Q - Another way to show the residuals compared to a normal distribution
4. Correlogram - ACF of the residuals, 95% of the residuals should not be significant (inside the shaded area)

print(results.summary())
# prob(Q) p-value for null hypothesis that residuals are uncorrelated (less than 0.05  we reject)
# prob(JB) p-value for null hypothesis that residuals are normal (less than 0.05  we reject)


##### Chapter 3
##### Section 3
##### Exercises

# Fit model
model = ARIMA(earthquake, order=(1,0,1))
results = model.fit()
# Calculate the mean absolute error from residuals
mae = np.mean(np.abs(results.resid))
# Print mean absolute error
print(mae)
# Make plot of time series for comparison
earthquake.plot()
plt.show()


# Create and fit model
model1 = ARIMA(df, order=(3,0,1))
results1 = model1.fit()
# Print summary
print(results1.summary())


# Create and fit model
model2 = ARIMA(df, order=(2,0,0))
results2 = model2.fit()

# Print summary
print(results2.summary()



# Fit model
model = ARIMA(earthquake, order=(1,0,1))
results = model.fit()

# Calculate the mean absolute error from residuals
mae = np.mean(np.abs(results.resid))

# Print mean absolute error
print(mae)

# Make plot of time series for comparison
earthquake.plot()
plt.show()


##### Chapter 3
##### Section 4
##### Box-Jenkins Method
"""
Box-Jenkins
The checklist for taking a model and going from raq data to something production worthy.

1. Identification:
    Is the time series stationary?
        Which transformations will make it stationary?
    What transforms will make it stationary?
    df.plot()
    adfuller()
    df.diff()
    np.log()
    np.sqrt()
    plot_acf()
    plot_pacf()

2. Estimation:
    Using numerical methods to estimate
    model.fit()
    results.aic()
    results.bic()

3. Model Diagnostics:
    results.plot_diagnostics()
    results.summary()

Once everything checks out we use the forecast.
results.get_forecast()
"""

##### Chapter 3
##### Section 4
##### Exercises

# Plot time series
savings.plot()
plt.show()
# Run Dicky-Fuller test
result = adfuller(savings['savings'])
# Print test statistic
print(result)


# Create figure
fig, (ax1, ax2) = plt.subplots(2,1, figsize=(12,8))
# Plot the ACF of savings on ax1
plot_acf(savings['savings'], lags=10, zero= False, ax=ax1)
# Plot the PACF of savings on ax2
plot_pacf(savings['savings'], lags=10, zero= False, ax=ax2)
plt.show()


# Loop over p values from 0-3
for p in range(4):
  # Loop over q values from 0-3
    for q in range(4):
      try:
        # Create and fit ARMA(p,q) model
        model = ARIMA(savings['savings'], order=(p,0,q))
        results = model.fit()
        
        # Print p, q, AIC, BIC
        print(p , q, results.aic, results.bic)
        
      except:
        print(p, q, None, None)


# Create and fit model
model = ARIMA(savings['savings'], order=(1,0,2))
results = model.fit()

# Create the 4 diagostics plots
results.plot_diagnostics()
plt.show()

# Print summary
print(results.summary())
