import numpy as np


def simple_exponential_smoothing(series, alpha):
    """
    Simple Exponential Smoothing (SES) algorithm for time series forecasting.

    Parameters:
    - series: 1D NumPy array representing the time series data.
    - alpha: Smoothing parameter (0 < alpha < 1).

    Returns:
    - predictions: NumPy array containing the forecasted values.
    """
    n = len(series)
    predictions = np.zeros(n)

    # Initialize the first prediction as the first data point
    predictions[0] = series[0]

    # Perform exponential smoothing
    for i in range(1, n):
        predictions[i] = alpha * series[i - 1] + (1 - alpha) * predictions[i - 1]

    return predictions


# Example usage:
# Generate a sample time series data
np.random.seed(42)
time_series_data = np.random.rand(50)

# Set smoothing parameter (alpha)
alpha = 0.2

# Apply simple exponential smoothing
forecasted_values = simple_exponential_smoothing(time_series_data, alpha)

# Print the results
print("Original Time Series:", time_series_data)
print("Forecasted Values:", forecasted_values)
