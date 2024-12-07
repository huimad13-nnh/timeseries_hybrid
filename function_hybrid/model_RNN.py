import numpy as np
from bayes_opt import BayesianOptimization
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError

# Map activation indices to functions
activation_map = {0: 'relu', 1: 'tanh', 2: 'sigmoid'}

def create_model_rnn(hidden_units = 50, activation = 'sigmoid', input_shape = (5,1), learning_rate = 0.01):
    model = Sequential()
    # Thêm Input layer
    model.add(Input(shape=input_shape))
    model.add(SimpleRNN(hidden_units, activation=activation))
    model.add(Dense(1))  # Single output node
    
    # Compile the model
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss=MeanSquaredError())
    return model

def find_best_params(X_train, y_train, validation_split=0.2):
    """
    Find the best hyperparameters for an RNN using Bayesian Optimization.
    
    Parameters:
        X_train (numpy.ndarray): Training data features (3D array: samples, time_steps, features).
        y_train (numpy.ndarray): Training data targets (2D array: samples, output_dim).
        validation_split (float): Fraction of training data to use for validation.
        
    Returns:
        dict: Best hyperparameters found during optimization.
    """
    
    def train_evaluate_rnn(hidden_units, learning_rate, batch_size, epochs, activation):
        """
        Train and evaluate an RNN with the given hyperparameters.
        
        Parameters:
            hidden_units (float): Number of hidden units (will be converted to int).
            learning_rate (float): Learning rate for the optimizer.
            batch_size (float): Batch size for training (will be converted to int).
            epochs (float): Number of epochs for training (will be converted to int).
            activation (float): Activation function index (mapped to a string).

        Returns:
            float: Negative validation loss (to maximize validation performance).
        """
        # Convert parameters to appropriate types
        hidden_units = int(hidden_units)
        batch_size = int(batch_size)
        epochs = int(epochs)
        activation = activation_map[int(activation)]
        
        # Build the RNN model (fixed to 1 layer)
        model = create_model_rnn(hidden_units=hidden_units, activation=activation, learning_rate=learning_rate)
        
        # Train the model
        history = model.fit(
            X_train, y_train,
            validation_split=validation_split,
            epochs=epochs,
            batch_size=batch_size,
            verbose=0
        )
        
        # Evaluate on validation set
        val_loss = history.history['val_loss'][-1]
        return -val_loss  # Return negative loss for maximization
    
    # Define the parameter space
    pbounds = {
        'hidden_units': (10, 100),          # Number of hidden units
        'learning_rate': (1e-4, 1e-2),      # Learning rate
        'batch_size': (16, 128),            # Batch size
        'epochs': (5, 50),                  # Number of epochs
        'activation': (0, 2)                # Activation: 0='relu', 1='tanh', 2='sigmoid'
    }
    
    # Bayesian Optimization
    optimizer = BayesianOptimization(
        f=train_evaluate_rnn,
        pbounds=pbounds,
        random_state=42,
        verbose=2
    )
    
    optimizer.maximize(init_points=5, n_iter=10)

    # Extract the best parameters
    best_params = optimizer.max['params']
    best_params['hidden_units'] = int(best_params['hidden_units'])
    best_params['epochs'] = int(best_params['epochs'])
    best_params['batch_size'] = int(best_params['batch_size'])
    best_params['activation'] = activation_map[int(best_params['activation'])]  # Map back to activation function

    return best_params

def train(X_train, y_train, hidden_units=50, activation='sigmoid', input_shape=(5,1), epochs=50, batch_size=32, learning_rate = 0.01):
    """
    Hàm này huấn luyện mô hình FFNN.
    """
    # Tạo mô hình
    model = create_model_rnn(hidden_units=hidden_units, activation=activation, input_shape=input_shape, learning_rate=learning_rate)
    
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

def fit_and_predict(X_train, y_train, X_test, hidden_units=32, activation='sigmoid', input_shape=(5,1), epochs=50, batch_size=32):
    """
    Kết hợp việc huấn luyện và dự đoán trong một hàm.
    """
    model, history = train(
        X_train,
        y_train,
        hidden_units=hidden_units,
        activation=activation,
        input_shape=input_shape,
        epochs=epochs,
        batch_size=batch_size
    )
    y_pred = predict(model, X_test)
    return model, history, y_pred

# Example usage:
# X_train, y_train = <your data here>
# best_params = find_best_params(X_train, y_train)
# print("Best parameters:", best_params)
