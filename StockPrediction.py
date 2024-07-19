#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 10:10:27 2024

@author: jamesturner
"""

import numpy as np
import pandas as pd
import yfinance as yf
import joblib
import matplotlib.pyplot as plt
from sklearn.utils import resample

# Set random seed for reproducibility
np.random.seed(42)

# Step 1: Collect New Data
ticker = 'JPM'
new_data = yf.download(ticker, start='2010-01-02', end='2023-07-01')
new_data.reset_index(inplace=True)

# Preprocessing
new_data['Date'] = pd.to_datetime(new_data['Date'])
new_data.set_index('Date', inplace=True)
new_data['MA_10'] = new_data['Adj Close'].rolling(window=10).mean()
new_data['MA_50'] = new_data['Adj Close'].rolling(window=50).mean()
new_data.dropna(inplace=True)

# Step 2: Load the trained model
model = joblib.load('stock_price_model.pkl')

# Step 3: Make Predictions
features = ['MA_10', 'MA_50']
X_new = new_data[features]
new_data['Predicted_Adj_Close'] = model.predict(X_new)

# Step 4: Forecasting Future Dates
future_dates = pd.date_range(start='2023-07-02', periods=1440, freq='D')
future_data = pd.DataFrame(index=future_dates)

# Initialize future moving averages with the last known values
last_ma_10 = new_data['MA_10'].iloc[-1]
last_ma_50 = new_data['MA_50'].iloc[-1]

# Simulate future 'Adj Close' values by using the predicted values
last_adj_close = new_data['Adj Close'].iloc[-1]
future_adj_close = [last_adj_close]

# Project future 'Adj Close' values using a simple model (e.g., random walk)
for i in range(1, len(future_dates)):
    future_adj_close.append(future_adj_close[-1] * (1 + np.random.normal(0, 0.01)))  # Assuming 1% daily volatility

# Convert list to DataFrame
future_data['Adj Close'] = future_adj_close

# Calculate future moving averages
future_data['MA_10'] = future_data['Adj Close'].rolling(window=10, min_periods=1).mean()
future_data['MA_50'] = future_data['Adj Close'].rolling(window=50, min_periods=1).mean()

# Predict future prices
X_future = future_data[features]
future_data['Predicted_Adj_Close'] = model.predict(X_future)

# Step 5: Bootstrapping for Confidence Intervals
n_iterations = 1000
predictions = np.zeros((n_iterations, len(future_data)))

for i in range(n_iterations):
    # Bootstrap sample
    X_resampled, y_resampled = resample(X_new, new_data['Adj Close'])
    
    # Train the model on the resampled data
    model.fit(X_resampled, y_resampled)
    
    # Predict future prices
    predictions[i, :] = model.predict(X_future)

# Calculate confidence intervals
lower_bound = np.percentile(predictions, 5, axis=0)
upper_bound = np.percentile(predictions, 95, axis=0)

# Step 6: Plot the results
plt.figure(figsize=(10, 5))
plt.plot(new_data.index, new_data['Adj Close'], label='Actual Prices')
plt.plot(future_data.index, future_data['Predicted_Adj_Close'], label='Predicted Prices')
plt.fill_between(future_data.index, lower_bound, upper_bound, color='gray', alpha=0.2, label='90% Confidence Interval')
plt.title('Actual vs Predicted Stock Prices with 90% Confidence Interval')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()