import tensorflow as tf
from keras import Sequential
from keras.layers import Dense, Input
from keras.optimizers import Adam
from bayes_opt import BayesianOptimization
from sklearn.model_selection import train_test_split

# Define the model creation function
def create_model_ffnn(hidden_layers=1, neurons=32, activation='relu', input_shape=(5,), learning_rate=1e-3):
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(Dense(neurons, activation=activation))
    
    for _ in range(hidden_layers - 1):
        model.add(Dense(neurons, activation=activation))
    
    model.add(Dense(1))  # Output layer
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    return model

# Define the Bayesian optimization objective function
def ffnn_objective(hidden_layers, neurons, learning_rate, batch_size, epochs):
    hidden_layers = int(hidden_layers)  # Convert to integer
    neurons = int(neurons)
    batch_size = int(batch_size)
    epochs = int(epochs)
    
    # Create and compile the model
    model = create_model_ffnn(
        hidden_layers=hidden_layers,
        neurons=neurons,
        activation='relu',
        input_shape=(X_train.shape[1],),
        learning_rate=learning_rate
    )
    
    # Train the model
    history = model.fit(
        X_train,
        y_train,
        validation_split=0.2,
        epochs=epochs,
        batch_size=batch_size,
        verbose=0
    )
    
    # Return the negative validation loss
    val_loss = history.history['val_loss'][-1]
    return -val_loss

# Perform Bayesian Optimization
def tune_hyperparameters(X_train, y_train):
    # Define parameter bounds
    pbounds = {
        'hidden_layers': (1, 3),      # Number of hidden layers
        'neurons': (16, 128),         # Number of neurons per layer
        'learning_rate': (1e-4, 1e-2), # Learning rate
        'batch_size': (16, 128),      # Batch size
        'epochs': (50, 150),          # Number of epochs
    }
    
    optimizer = BayesianOptimization(
        f=ffnn_objective,
        pbounds=pbounds,
        random_state=42,
    )
    
    optimizer.maximize(init_points=5, n_iter=25)
    
    return optimizer.max

# Train the final model using the best hyperparameters
def train_final_model(X_train, y_train, best_params):
    model = create_model_ffnn(
        hidden_layers=int(best_params['params']['hidden_layers']),
        neurons=int(best_params['params']['neurons']),
        activation='relu',
        input_shape=(X_train.shape[1],),
        learning_rate=best_params['params']['learning_rate']
    )
    
    history = model.fit(
        X_train,
        y_train,
        validation_split=0.2,
        epochs=int(best_params['params']['epochs']),
        batch_size=int(best_params['params']['batch_size']),
        verbose=1
    )
    
    return model, history

# Example usage
if __name__ == "__main__":
    # Replace this with loading your actual dataset
    X, y = # Your dataset goes here
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Tune hyperparameters
    best_params = tune_hyperparameters(X_train, y_train)
    print("Best hyperparameters:", best_params)
    
    # Train the final model
    final_model, history = train_final_model(X_train, y_train, best_params)
