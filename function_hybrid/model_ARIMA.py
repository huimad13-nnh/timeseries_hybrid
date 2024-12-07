import pmdarima as pm
from statsmodels.tsa.arima.model import ARIMA
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning
import numpy as np

def train_arima_model(train):
    """
    Train an ARIMA model by finding the optimal (p, d, q) parameters using auto_arima.

    Parameters:
    ----------
    train : array-like
        The training dataset.

    Returns:
    -------
    arima_model : pmdarima.ARIMA
        The fitted ARIMA model containing optimal parameters.
    order : tuple
        The optimal (p, d, q) order for the ARIMA model.
    """
    arima_model = pm.auto_arima(
        train,
        start_p=1, start_q=1,
        max_p=7, max_q=7,
        start_d=0, max_d=5,
        seasonal=False,
        trace=True,
        stepwise=True
    )
    order = arima_model.order
    print(f'Optimal order: p={order[0]}, d={order[1]}, q={order[2]}')
    return arima_model, order


def fit_arima_model(train, order):
    """
    Fit an ARIMA model with the specified order on the training dataset.

    Parameters:
    ----------
    train : array-like
        The training dataset.
    order : tuple
        The (p, d, q) order for the ARIMA model.

    Returns:
    -------
    fitted_model : statsmodels.tsa.arima.model.ARIMAResults
        The fitted ARIMA model.
    """
    try:
        model = ARIMA(train, order=order)
        fitted_model = model.fit()
        return fitted_model
    except ConvergenceWarning:
        print("Warning: Model did not converge. Check data or model parameters.")
        return None


def forecast_arima(train, test, order):
    """
    Generate predictions for both the training and test datasets using ARIMA.

    Parameters:
    ----------
    train : array-like
        The training dataset.
    test : array-like
        The test dataset.
    order : tuple
        The (p, d, q) order for the ARIMA model.

    Returns:
    -------
    train_predictions : array-like
        Predictions for the training data (excluding the first 10 points).
    test_predictions : numpy.ndarray
        Predictions for the test dataset generated iteratively.
    """
    # Fit the model on the training data
    fitted_model = fit_arima_model(train, order)

    # Predict for train[10:]
    start_index = 10
    end_index = len(train) - 1
    train_predictions = fitted_model.predict(start=start_index, end=end_index)
    
    # Iterative forecasting for the test set
    history = list(train)
    test_predictions = []

    for t in range(len(test)):
        model = ARIMA(history, order=order)
        model_fit = model.fit()
        forecast = model_fit.forecast()
        yhat = forecast[0]
        test_predictions.append(yhat)
        history.append(test[t])

    return train[start_index:], test_predictions, train_predictions

# Examle:

# # Generate synthetic time series data
# np.random.seed(42)
# train_data = np.cumsum(np.random.normal(0, 1, 100))  # Training data: cumulative sum of random noise
# test_data = np.cumsum(np.random.normal(0, 1, 20))    # Test data: cumulative sum of random noise

# # Step 1: Train ARIMA model and find optimal order
# arima_model, optimal_order = train_arima_model(train_data)

# # Step 2: Generate predictions for training and test datasets
# train_segment, test_predictions = forecast_arima(train_data, test_data, optimal_order)