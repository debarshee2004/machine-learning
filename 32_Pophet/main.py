import numpy as np
import matplotlib.pyplot as plt

# Generate synthetic time series data
np.random.seed(42)
date_rng = pd.date_range(start="2022-01-01", end="2022-12-31", freq="D")
y = np.cumsum(np.random.randn(len(date_rng)))

# Split the data into training and testing sets
train_size = int(len(y) * 0.8)
train, test = y[:train_size], y[train_size:]


# Function to fit the trend using moving averages
def fit_trend(y, window=7):
    trend = np.convolve(y, np.ones(window) / window, mode="valid")
    return np.concatenate([np.full(window - 1, np.nan), trend])


# Function to fit seasonality using weekly and yearly components
def fit_seasonality(y, weekly_window=7, yearly_window=365):
    weekly_season = np.array([np.nanmean(y[i::7]) for i in range(7)])
    yearly_season = np.array([np.nanmean(y[i::365]) for i in range(365)])

    seasonal_components = (
        np.tile(weekly_season, len(y) // 7 + 1)[: len(y)]
        + np.tile(yearly_season, len(y) // 365 + 1)[: len(y)]
    )
    return seasonal_components


# Function to forecast using the fitted trend and seasonality
def forecast(y, trend, seasonality):
    return trend + seasonality


# Fit trend and seasonality on training data
trend = fit_trend(train)
seasonality = fit_seasonality(train)

# Make predictions on test data
predictions = forecast(train, trend, seasonality)

# Plot the results
plt.figure(figsize=(12, 6))
plt.plot(date_rng[: len(train)], train, label="Training Data")
plt.plot(date_rng[len(train) :], test, label="True Test Data")
plt.plot(date_rng[: len(train)], predictions, label="Predictions", linestyle="dashed")
plt.title("Prophet Algorithm - Time Series Forecasting")
plt.xlabel("Date")
plt.ylabel("Value")
plt.legend()
plt.show()
