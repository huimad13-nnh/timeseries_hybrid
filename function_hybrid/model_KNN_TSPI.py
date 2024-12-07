import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# 1. Generate Synthetic Time Series Data
np.random.seed(42)
time = np.arange(0, 100, 0.1)
series = np.sin(time) + 0.5 * np.random.normal(size=len(time))

# 2. Z-Normalization Function
def z_normalize(series):
    scaler = StandardScaler()
    return scaler.fit_transform(series.reshape(-1, 1)).flatten(), scaler

# 3. Create Sliding Windows
def create_sliding_windows(series, window_size):
    windows = np.lib.stride_tricks.sliding_window_view(series, window_size)[:-1]
    targets = series[window_size:]
    return windows, targets

# 4. Dynamic Distance Metric
def calculate_distance(query, windows, metric="euclidean"):
    """
    Calculate distances between a query and all windows using the specified metric.
    Supports Euclidean, Manhattan, Chebyshev, Correlation, Cosine, and DTW metrics.
    """
    if metric == "euclidean":
        return cdist([query], windows, metric="euclidean").flatten()
    elif metric == "manhattan":
        return cdist([query], windows, metric="cityblock").flatten()
    elif metric == "chebyshev":
        return cdist([query], windows, metric="chebyshev").flatten()
    elif metric == "correlation":
        return cdist([query], windows, metric="correlation").flatten()
    elif metric == "cosine":
        return cdist([query], windows, metric="cosine").flatten()
    elif metric == "dtw":
        from dtaidistance import dtw
        return np.array([dtw.distance(query, w) for w in windows])
    else:
        raise ValueError(f"Unsupported distance metric: {metric}")


# 5. k-Nearest Neighbors Search
def find_k_nearest_neighbors(windows, query, k, metric="euclidean"):
    distances = calculate_distance(query, windows, metric=metric)
    indices = np.argsort(distances)[:k]
    return indices

# 6. Parallelized kNN Prediction
def knn_predict_parallel(series, window_size, k, metric="euclidean", n_jobs=-1):
    windows, targets = create_sliding_windows(series, window_size)

    def predict_single(query):
        neighbors_idx = find_k_nearest_neighbors(windows, query, k, metric=metric)
        neighbor_targets = targets[neighbors_idx]
        return np.mean(neighbor_targets)

    predictions = Parallel(n_jobs=n_jobs)(delayed(predict_single)(series[i:i + window_size])
                                          for i in range(len(series) - window_size))
    return predictions

# 7. Dynamic k Selection
def select_best_k(X_train, y_train, X_test, y_test , metric="euclidean"):
    
    best_k, best_mse = None, float("inf")
    k_values = range(1, 11)
    for k in k_values:
        predictions = []
        for i in range(len(X_test)):
            neighbors_idx = find_k_nearest_neighbors(X_train, X_test[i], k, metric=metric)
            neighbor_targets = y_train[neighbors_idx]
            predictions.append(np.mean(neighbor_targets))
        mse = np.mean((np.array(predictions) - y_test) ** 2)
        if mse < best_mse:
            best_k, best_mse = k, mse
    return best_k

def tune_best_distance(X_train, y_train, X_test, y_test, n_jobs=-1):
    """
    Tune the best distance metric by evaluating different metrics on a validation set.
    
    Parameters:
        series (array): Time series data.
        window_size (int): Length of the sliding window.
        k (int): Number of nearest neighbors.
        metrics (list): List of distance metrics to evaluate.
        validation_split (float): Fraction of data to use for validation.
        n_jobs (int): Number of parallel jobs for kNN prediction.
    
    Returns:
        best_metric (str): Best distance metric.
        errors (dict): Dictionary of validation errors for each metric.
    """    
    errors = {}
    best_k = {}
    metrics = ["euclidean", "manhattan", "chebyshev", "correlation", "cosine", "dtw"]
    for metric in metrics:
        k = select_best_k(X_train, y_train, X_test, y_test)
        predictions = []
        for i in range(len(X_test)):
            query = X_test[i]
            neighbors_idx = find_k_nearest_neighbors(X_train, query, k, metric=metric)
            neighbor_targets = y_train[neighbors_idx]
            prediction = np.mean(neighbor_targets)
            predictions.append(prediction)
        best_k[metric] = k
        # Calculate Mean Squared Error
        mse = np.mean((np.array(predictions) - y_test) ** 2)
        errors[metric] = mse

    # Select the best metric
    best_metric = min(errors, key=errors.get)
    return best_metric, errors, best_k

def knn_predict(X_train, y_train, X_test, k, metric="euclidean", n_jobs=-1):
    """
    Predict values for X_test using k-Nearest Neighbors on the provided training set.
    
    Parameters:
        X_train (array): Sliding windows from training data (features).
        y_train (array): Target values corresponding to X_train.
        X_test (array): Sliding windows from test data (features).
        k (int): Number of nearest neighbors.
        metric (str): Distance metric for similarity computation.
        n_jobs (int): Number of parallel jobs.
        
    Returns:
        predictions (list): Predicted values for X_test.
    """
    # Parallel prediction for test data
    def predict_single(query):
        neighbors_idx = find_k_nearest_neighbors(X_train, query, k, metric=metric)
        neighbor_targets = y_train[neighbors_idx]
        return np.mean(neighbor_targets)
    
    predictions = Parallel(n_jobs=n_jobs)(delayed(predict_single)(query) for query in X_test)
    return np.array(predictions)
