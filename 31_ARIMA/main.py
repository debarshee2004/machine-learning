import numpy as np


def difference(dataset, interval=1):
    diff = list()
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]
        diff.append(value)
    return np.array(diff)


def inverse_difference(history, yhat, interval=1):
    return yhat + history[-interval]


def fit_ARIMA(series, p, d, q):
    # Make series stationary by differencing
    stationary = series
    for i in range(d):
        stationary = difference(stationary)

    # Fit AR and MA components
    ar_coef = np.polyfit(stationary, np.roll(stationary, shift=-1), deg=p)
    ma_coef = np.polyfit(stationary, np.roll(stationary, shift=-1), deg=q)

    return ar_coef, ma_coef


def forecast_ARIMA(series, ar_coef, ma_coef, n_steps):
    history = list(series)
    predictions = list()

    for _ in range(n_steps):
        # Make series stationary by differencing
        diff = difference(history)

        # Forecast using AR and MA components
        ar_value = np.polyval(ar_coef, diff[-1])
        ma_value = np.polyval(ma_coef, diff[-1])

        # Invert differencing
        yhat = inverse_difference(history, ar_value + ma_value, 1)
        predictions.append(yhat)

        # Update history
        history.append(yhat)

    return np.array(predictions)


# Example usage
np.random.seed(42)
# Generate a random time series data
time_series = np.cumsum(np.random.normal(0, 1, 100))

# Set ARIMA hyperparameters (p, d, q)
p = 1  # AR order
d = 1  # Integration order
q = 1  # MA order

# Fit ARIMA model
ar_coef, ma_coef = fit_ARIMA(time_series, p, d, q)

# Forecast future values
n_steps = 10
predictions = forecast_ARIMA(time_series, ar_coef, ma_coef, n_steps)

print("Original time series:", time_series)
print("ARIMA Predictions:", predictions)
