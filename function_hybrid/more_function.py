import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.metrics import mean_absolute_error, mean_squared_error

def calculate_mape(y_true, y_pred):
    """
    Calculate Mean Absolute Percentage Error (MAPE).

    Parameters:
        y_true (array-like): Array of true values.
        y_pred (array-like): Array of predicted values.

    Returns:
        float: MAPE as a percentage.
    """
    # Ensure inputs are numpy arrays for vectorized operations
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Validate that the arrays have the same shape
    if y_true.shape != y_pred.shape:
        raise ValueError("Input arrays must have the same shape.")
    
    # Add small epsilon to avoid division by zero
    epsilon = 1e-10
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100

    return mape

def compare_multiple_models(y_true, predictions, model_names):
    """
    So sánh các chỉ số lỗi giữa nhiều mô hình.
    
    Parameters:
        y_true: array-like
            Giá trị thực tế.
        predictions: list of array-like
            Danh sách các giá trị dự đoán từ các mô hình.
        model_names: list of str
            Danh sách tên các mô hình.
    
    Returns:
        Pandas DataFrame
            Bảng so sánh các chỉ số lỗi giữa các mô hình.
    """
    metrics = {"Metric": ["Mean Squared Error (MSE)", "Mean Absolute Error (MAE)", "Mean Absolute Percentage Error (MAPE)"]}
    for pred, name in zip(predictions, model_names):
        metrics[name] = [
            mean_squared_error(y_true, pred),
            mean_absolute_error(y_true, pred),
            calculate_mape(y_true, pred)
        ]
    
    return pd.DataFrame(metrics)

def calculate_residuals(y_actual, y_predicted):
    """
    Calculate residuals between actual and predicted series.

    Parameters:
    - y_actual (pd.Series or np.ndarray): Actual observed values.
    - y_predicted (pd.Series or np.ndarray): Predicted values from the ARIMA model.

    Returns:
    - residuals (pd.Series): Residuals as the difference between actual and predicted values.
    """

    # Ensure the series are of the same length
    if len(y_actual) != len(y_predicted):
        raise ValueError("The length of y_actual and y_predicted must be the same.")

    # Calculate residuals
    residuals = y_actual - y_predicted
    
    # Return as a pandas Series
    return residuals


def calculate_metrics(y_true, y_pred):
    """
    Calculate MSE, MAE, and MAPE between true and predicted values.

    Parameters:
        y_true (array-like): Array of true values.
        y_pred (array-like): Array of predicted values.

    Returns:
        dict: A dictionary containing MSE, MAE, and MAPE.
    """
    # Ensure inputs are numpy arrays for vectorized operations
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Validate that the arrays have the same shape
    if y_true.shape != y_pred.shape:
        raise ValueError("Input arrays must have the same shape.")
    
    # Mean Squared Error
    mse = np.mean((y_true - y_pred) ** 2)
    
    # Mean Absolute Error
    mae = np.mean(np.abs(y_true - y_pred))
    
    # Mean Absolute Percentage Error
    mape = calculate_mape(y_true, y_pred)

    return {"MSE": mse, "MAE": mae, "MAPE": mape}

def variance(data):
    """
    Compute the variance of a dataset.

    Parameters:
    ----------
    data : array-like
        A sequence of numerical data for which the variance is to be computed.

    Returns:
    -------
    float
        The variance of the dataset, calculated as the average of the squared differences
        from the mean.

    Example:
    -------
    >>> data = [1, 2, 3, 4]
    >>> variance(data)
    1.25
    """
    mean = np.mean(data)
    return np.mean((data - mean) ** 2)


def covariance(data1, data2):
    """
    Compute the covariance between two datasets.

    Parameters:
    ----------
    data1 : array-like
        The first dataset, a sequence of numerical data.
    data2 : array-like
        The second dataset, a sequence of numerical data.

    Returns:
    -------
    float
        The covariance between `data1` and `data2`, calculated as the average product of 
        deviations from their respective means.

    Example:
    -------
    >>> data1 = [1, 2, 3]
    >>> data2 = [4, 5, 6]
    >>> covariance(data1, data2)
    1.0
    """
    mean1, mean2 = np.mean(data1), np.mean(data2)
    return np.mean((data1 - mean1) * (data2 - mean2))

# Function to calculate weights
def calculate_weights(e1, e2):
    var_e1 = variance(e1)
    var_e2 = variance(e2)
    cov_e1_e2 = covariance(e1, e2)
    
    # Calculate weights
    w1 = (var_e2 - cov_e1_e2) / (var_e1 + var_e2 - 2 * cov_e1_e2)
    w2 = 1 - w1
    
    return w1, w2

def calculate_weights_from_variance(e1, e2):
    """
    Calculate weights based on squared errors.

    Parameters:
    ----------
    e1 : array-like
        Errors from the first forecast model.
    e2 : array-like
        Errors from the second forecast model.

    Returns:
    -------
    w1 : float
        Weight for the first forecast model.
    w2 : float
        Weight for the second forecast model.
    """
    # Compute variances (squared errors)
    e1_squared = np.var(e1)
    e2_squared = np.var(e2)
    
    # Calculate weights
    w1 = e2_squared / (e1_squared + e2_squared)
    w2 = e1_squared / (e1_squared + e2_squared)
    
    return w1, w2

def combined_forecast(f1, f2, w1, w2):
    """
    Calculate the combined forecast using weighted averages of two forecasts.

    Parameters:
    ----------
    f1 : array-like
        Forecast values from the first model.
    f2 : array-like
        Forecast values from the second model.
    w1 : float
        Weight assigned to the first forecast model.
    w2 : float
        Weight assigned to the second forecast model.

    Returns:
    -------
    array-like
        The combined forecast values.

    Notes:
    -----
    The combined forecast is calculated as:
        combined_forecast = w1 * f1 + w2 * f2

    Example:
    -------
    >>> f1 = [10, 20, 30]
    >>> f2 = [15, 25, 35]
    >>> w1, w2 = 0.6, 0.4
    >>> combined_forecast(f1, f2, w1, w2)
    [12.0, 22.0, 32.0]
    """
    return w1 * f1 + w2 * f2


# Generalized function for combined forecasting
def combined_forecasting(y_actual, f1, f2):
    """
    Calculate the combined forecast, along with weights w1 and w2.
    
    Parameters:
    y_actual: np.array - Actual values
    f1: np.array - Forecasts from model 1
    f2: np.array - Forecasts from model 2
    
    Returns:
    w1, w2: float - Weights for model 1 and model 2
    y_combined: np.array - Combined forecast
    """
    # Calculate errors
    e1 = y_actual - f1
    e2 = y_actual - f2
    
    # Calculate weights

    w1, w2 = calculate_weights(e1, e2)
    if (w1 < 0 or w2 < 0):
        w1, w2 = calculate_weights_from_variance(e1, e2)

    # Calculate combined forecast
    y_combined = combined_forecast(f1, f2, w1, w2)
    
    return w1, w2, y_combined

def plot_actual_vs_predicted(dates, actual, predicted):
    """
    Plot actual vs predicted values.

    Parameters:
        dates (array-like): List of time points (e.g., months or years).
        actual (array-like): List of actual values.
        predicted (array-like): List of predicted values.

    Returns:
        None: Displays the plot.
    """
    plt.figure(figsize=(10, 6))
    
    # Plot actual values
    plt.plot(dates, actual, label="Actual Values", color="blue", linewidth=2)
    
    # Plot predicted values
    plt.plot(dates, predicted, label="Predicted Values", color="orange", linestyle="--", linewidth=2)
    
    # Add title and labels
    plt.title("Actual vs Predicted Values", fontsize=16)
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Value", fontsize=12)
    
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3))  # Show every 3rd month
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))  # Format as 'Month Year'

    # Add legend
    plt.legend(fontsize=12)
    
    # Improve date formatting
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    # Show the plot
    plt.show()