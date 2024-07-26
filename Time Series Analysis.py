#!/usr/bin/env python
# coding: utf-8

# In[4]:


# Install necessary libraries
get_ipython().system('pip install yfinance statsmodels pmdarima fbprophet matplotlib seaborn')


# In[7]:


import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error


# In[8]:


# Define the stock ticker and time range
ticker = 'AAPL'  # Example: Apple Inc.
start_date = '2010-01-01'
end_date = '2023-01-01'

# Download stock price data
data = yf.download(ticker, start=start_date, end=end_date)
data = data['Close']

# Display the first few rows of the dataset
data.head()


# In[9]:


# Plot historical stock prices
plt.figure(figsize=(12, 6))
sns.lineplot(data=data, label='Stock Price')
plt.title(f'{ticker} Stock Prices')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()

# Check for stationarity
result = adfuller(data.dropna())
print('ADF Statistic:', result[0])
print('p-value:', result[1])


# In[10]:


# Split data into training and testing sets
train = data[:-30]
test = data[-30:]

# Fit ARIMA model
arima_model = ARIMA(train, order=(5,1,0))
arima_fit = arima_model.fit()

# Forecast
arima_forecast = arima_fit.forecast(steps=30)
arima_forecast_series = pd.Series(arima_forecast, index=test.index)

# Plot ARIMA forecast
plt.figure(figsize=(12, 6))
plt.plot(train, label='Train Data')
plt.plot(test, label='Test Data')
plt.plot(arima_forecast_series, label='ARIMA Forecast', color='red')
plt.title('ARIMA Forecast')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()


# In[11]:


# Fit Exponential Smoothing model
ets_model = ExponentialSmoothing(train, trend='add', seasonal='add', seasonal_periods=12)
ets_fit = ets_model.fit()

# Forecast
ets_forecast = ets_fit.forecast(steps=30)
ets_forecast_series = pd.Series(ets_forecast, index=test.index)

# Plot Exponential Smoothing forecast
plt.figure(figsize=(12, 6))
plt.plot(train, label='Train Data')
plt.plot(test, label='Test Data')
plt.plot(ets_forecast_series, label='ETS Forecast', color='green')
plt.title('Exponential Smoothing Forecast')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()


# In[13]:


print(test.isna().sum())
print(arima_forecast_series.isna().sum())
print(ets_forecast_series.isna().sum())


# In[14]:


test = test.dropna()
arima_forecast_series = arima_forecast_series.dropna()
ets_forecast_series = ets_forecast_series.dropna()


# In[15]:


test = test.fillna(method='ffill')  # Forward fill
arima_forecast_series = arima_forecast_series.fillna(method='ffill')
ets_forecast_series = ets_forecast_series.fillna(method='ffill')


# In[17]:


print(f'Length of test: {len(test)}')
print(f'Length of arima_forecast_series: {len(arima_forecast_series)}')
print(f'Length of ets_forecast_series: {len(ets_forecast_series)}')


# In[18]:


print(test.isna().sum())
print(arima_forecast_series.isna().sum())
print(ets_forecast_series.isna().sum())


# In[19]:


min_length = min(len(test), len(arima_forecast_series), len(ets_forecast_series))
test = test[:min_length]
arima_forecast_series = arima_forecast_series[:min_length]
ets_forecast_series = ets_forecast_series[:min_length]


# In[21]:


print(f'Length of test: {len(test)}')
print(f'Length of arima_forecast_series: {len(arima_forecast_series)}')
print(f'Length of ets_forecast_series: {len(ets_forecast_series)}')


# In[22]:


# Example code to inspect and handle empty arrays
if len(test) == 0 or len(arima_forecast_series) == 0:
    print("One of the arrays is empty. Please check the data generation process.")
else:
    mae_arima = mean_absolute_error(test, arima_forecast_series)
    mape_arima = mean_absolute_percentage_error(test, arima_forecast_series)
    print(f'ARIMA MAE: {mae_arima:.2f}, MAPE: {mape_arima:.2f}')

if len(test) == 0 or len(ets_forecast_series) == 0:
    print("One of the arrays is empty. Please check the data generation process.")
else:
    mae_ets = mean_absolute_error(test, ets_forecast_series)
    mape_ets = mean_absolute_percentage_error(test, ets_forecast_series)
    print(f'ETS MAE: {mae_ets:.2f}, MAPE: {mape_ets:.2f}')


# In[23]:


# Plot forecasted vs. actual prices
plt.figure(figsize=(12, 6))
plt.plot(data, label='Historical Data')
plt.plot(arima_forecast_series, label='ARIMA Forecast', color='red')
plt.plot(ets_forecast_series, label='ETS Forecast', color='green')
plt.title('Forecasted vs. Actual Stock Prices')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()


# In[ ]:




