from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV

def find_best_params(X_train, y_train, cv=5, scoring='neg_mean_squared_error', verbose=2):
    """
    Perform grid search to find the best hyperparameters for an SVR model.

    Parameters:
    ----------
    X_train : np.ndarray
        Training feature matrix.
    y_train : np.ndarray
        Training target vector.
    param_grid : dict
        Dictionary specifying the parameter grid to search.
    cv : int, optional
        Number of cross-validation folds (default is 5).
    scoring : str, optional
        Scoring metric to optimize (default is 'neg_mean_squared_error').
    verbose : int, optional
        Verbosity level for grid search output (default is 2).

    Returns:
    -------
    dict
        Dictionary containing the best parameters and the best cross-validation score.
    """
    # Ensure X_train is in the correct shape for the model
    X_train = X_train.reshape(X_train.shape[0], -1)

    param_grid = {
        'C': [0.1, 1, 10, 100],
        'epsilon': [0.01, 0.05, 0.1, 0.5],
        'gamma': ['scale', 'auto', 0.1, 0.5, 1],
        'kernel': ['linear', 'rbf', 'sigmoid']
    }

    # Initialize the SVR model and GridSearchCV
    grid_search = GridSearchCV(
        estimator=SVR(),
        param_grid=param_grid,
        scoring=scoring,
        cv=cv,
        verbose=verbose,
        n_jobs=-1
    )

    # Perform the grid search
    grid_search.fit(X_train, y_train)

    # Extract best parameters and score
    best_params = grid_search.best_params_

    return best_params

def train(X_train, y_train, c = 0.1, epsilon = 0.01, gamma = 'scale', kernel = 'linear'):
    model = SVR(C=c, epsilon=epsilon, gamma= gamma, kernel=kernel)

    history = model.fit(
        X_train,
        y_train
    )

    return model, history

def predict(model, X_test):
    pred = model.predict(X_test)
    return pred

def fit_and_predict(X_train, y_train, X_test, c = 0.1, epsilon = 0.01, gamma = 'scale', kernel = 'linear'):
    """
    Kết hợp việc huấn luyện và dự đoán trong một hàm.
    """
    model, history = train(
        X_train,
        y_train,
        c=c,
        epsilon=epsilon,
        gamma= gamma,
        kernel=kernel
    )
    
    train_pred = predict(model, X_train)
    test_pred = predict(model, X_test)
    return model, history, test_pred, train_pred