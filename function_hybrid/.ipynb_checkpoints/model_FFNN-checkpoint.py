import tensorflow as tf
import keras
from keras import layers
from keras import models
from keras import utils
from keras.models import Sequential
from keras.layers import Dense, Input
from scikeras.wrappers import KerasRegressor
from sklearn.model_selection import GridSearchCV
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping


# Define the model creation function
def create_model_ffnn(hidden_layers=1, neurons=50, activation='relu', input_shape=(5,)):
    model = Sequential()
    # Thêm Input layer
    model.add(Input(shape=input_shape))
    model.add(Dense(neurons, activation=activation))
    
    for _ in range(hidden_layers - 1):
        model.add(Dense(neurons, activation=activation))
    
    model.add(Dense(1))  # Lớp đầu ra
    optimizer = Adam()  
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    return model

# Grid search function
def find_bestparam(X_train, y_train, X_test, y_test):  
    # Tạo mô hình  
    model = KerasRegressor(model=create_model_ffnn, input_shape=(X_train.shape[1],))  
    
    param_grid = {
        'model__hidden_layers': [1, 2, 3],
        'model__neurons': [16, 32, 64, 128],
        'model__activation': ['relu', 'sigmoid', 'tanh'],
        'batch_size': [16, 32, 64, 128],      # Batch size for KerasRegressor
        'epochs': [50, 100, 150]
    }
    
    # Perform GridSearchCV for each time_step
    grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, scoring='neg_mean_squared_error', verbose=1, n_jobs = -1)
    grid_result = grid.fit(X_train, y_train)

    return grid_result.best_params_

def train(X_train, y_train, num_layers_hidden=1, neuralHidden=32, activation='relu', input_shape=(5,), epochs=50, batch_size=32):
    """
    Hàm này huấn luyện mô hình FFNN.
    """
    # Tạo mô hình
    model = create_model_FFNN(
        num_layers_hidden=num_layers_hidden,
        neuralHidden=neuralHidden,
        activation=activation,
        input_shape=input_shape
    )
    
    # Huấn luyện mô hình
    history = model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,  # Phân chia dữ liệu để kiểm tra
        verbose=1
    )
    
    return model, history


def predict(model, X_test):
    """
    Hàm này thực hiện dự đoán dựa trên mô hình đã được huấn luyện.
    """
    y_pred = model.predict(X_test)
    return y_pred


def fit_and_predict(X_train, y_train, X_test, num_layers_hidden=1, neuralHidden=32, activation='relu', input_shape=(5,), epochs=50, batch_size=32):
    """
    Kết hợp việc huấn luyện và dự đoán trong một hàm.
    """
    model, history = train(
        X_train,
        y_train,
        num_layers_hidden=num_layers_hidden,
        neuralHidden=neuralHidden,
        activation=activation,
        input_shape=input_shape,
        epochs=epochs,
        batch_size=batch_size
    )
    
    y_pred = predict(model, X_test)
    return model, history, y_pred

