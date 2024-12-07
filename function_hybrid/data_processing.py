# data_processing.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def read_data(file_path, cols, date_col):
    """Load data from CSV."""
    data_csv = pd.read_csv(file_path)
    data = data_csv[[cols]]
    date = data_csv[[date_col]]
    return data, date

def clean_data(data, cols, scale=True):
    """Clean and preprocess the data.

    Parameters:
    - data (DataFrame): The input dataset.
    - cols (list): List of columns to process.
    - scale (bool): Whether to apply MinMax scaling. Default is True.

    Returns:
    - DataFrame: Cleaned data.
    - scaler (MinMaxScaler, optional): The scaler object if scaling is applied.
    """
    data.fillna(0, inplace=True)  # Handle nulls

    if scale:
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        data[cols] = scaler.fit_transform(data[cols].values.reshape(-1, 1))
        data[cols] = data[cols].interpolate(method='linear')  # Interpolate missing
        return data, scaler
    else:
        data[cols] = data[cols].interpolate(method='linear')  # Interpolate missing
        return data
    

def split_data(data, percent_train):
    """Split data into train and test sets."""
    train_size = int(len(data) * (percent_train / 100))
    train_data = data.iloc[:train_size, :]
    test_data = data.iloc[train_size:, :]
    return train_data, test_data

def prepare_data(data, size_window, size_predict, step_window):
    """Prepare data for the neural network by creating sliding windows."""
    X, y = [], []
    start_window = 0
    for i in range(len(data) - size_window - 1):
        if len(data[(start_window + size_window):(start_window + size_window + size_predict)]) != size_predict:
            break
        X.append(data[start_window:(start_window + size_window):])
        y.append(data[(start_window + size_window):(start_window + size_window + size_predict)])
        start_window += step_window
    return np.array(X), np.array(y)
