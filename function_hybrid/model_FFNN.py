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
from bayes_opt import BayesianOptimization

# Define the model creation function
def create_model_ffnn(hidden_layers=1, neurons=32, activation='relu', input_shape=(5,), learning_rate = 0.001):
    model = Sequential()
    # Thêm Input layer
    model.add(Input(shape=input_shape))
    model.add(Dense(neurons, activation=activation))
    
    for _ in range(hidden_layers - 1):
        model.add(Dense(neurons, activation=activation))
    
    model.add(Dense(1))  # Lớp đầu ra
    optimizer = Adam(learning_rate=learning_rate)  
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    return model

# Mapping for activation functions
activation_mapping = {0: 'relu', 1: 'sigmoid', 2: 'tanh'}

# Bayesian Optimization-based parameter tuning
def find_bestparam_bayesian(X_train, y_train, input_shape=(5,)):
    """
    Use Bayesian Optimization to find the best hyperparameters for FFNN, including activation function.
    """
    def objective_function(hidden_layers, neurons, learning_rate, epochs, batch_size, activation):
        # Convert parameters to integer where necessary
        hidden_layers = int(hidden_layers)
        neurons = int(neurons)
        epochs = int(epochs)
        batch_size = int(batch_size)
        activation = activation_mapping[int(activation)]  # Map activation index to function name

        # Create and compile the model
        model = create_model_ffnn(
            hidden_layers=hidden_layers,
            neurons=neurons,
            activation=activation,
            input_shape=input_shape
        )
        optimizer = Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss='mean_squared_error')

        # Train the model
        history = model.fit(
            X_train,
            y_train,
            validation_split=0.2,  # Use a portion of the training data for validation
            epochs=epochs,
            batch_size=batch_size,
            verbose=0
        )

        # Get the final validation loss
        val_loss = history.history['val_loss'][-1]

        # Return the negative validation loss (Bayesian Optimization maximizes the function)
        return -val_loss

    # Define the search space
    pbounds = {
        'hidden_layers': (1, 3),       # Number of hidden layers
        'neurons': (16, 128),          # Number of neurons in each layer
        'learning_rate': (1e-4, 1e-2), # Learning rate
        'epochs': (10, 100),           # Number of epochs
        'batch_size': (16, 128),       # Batch size
        'activation': (0, 2)           # Categorical: 0 = relu, 1 = sigmoid, 2 = tanh
    }

    # Perform Bayesian Optimization
    optimizer = BayesianOptimization(
        f=objective_function,
        pbounds=pbounds,
        random_state=42
    )

    optimizer.maximize(init_points=5, n_iter=25)

    # Extract the best parameters
    best_params = optimizer.max['params']
    best_params['hidden_layers'] = int(best_params['hidden_layers'])  # Convert to integer
    best_params['neurons'] = int(best_params['neurons'])
    best_params['epochs'] = int(best_params['epochs'])
    best_params['batch_size'] = int(best_params['batch_size'])
    best_params['activation'] = activation_mapping[int(best_params['activation'])]  # Map back to activation function

    return best_params



def train(X_train, y_train, num_layers_hidden=1, neuralHidden=32, activation='relu', input_shape=(5,), epochs=50, batch_size=32, learning_rate = 0.01):
    """
    Hàm này huấn luyện mô hình FFNN.
    """
    # Tạo mô hình
    model = create_model_ffnn(
        hidden_layers=num_layers_hidden,
        neurons=neuralHidden,
        activation=activation,
        input_shape=input_shape,
        learning_rate= learning_rate
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

