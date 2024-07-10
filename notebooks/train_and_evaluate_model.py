# Train and evaluation model function used in models.ipynb

import numpy as np
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error

def train_and_evaluate_model(model, model_name, X_train, y_train):
    """
    Train and evaluate a regression model using cross-validation.

    Parameters:
    - model: The regression model to train and evaluate.
    - model_name (str): Name of the regression model.
    - X_train (numpy.ndarray or pandas.DataFrame): Features of the training set.
    - y_train (numpy.ndarray or pandas.Series): Target variable of the training set.

    Returns:
    None

    Prints:
    - Mean Absolute Error (MSE) on the training set.
    - Mean Absolute Error (MAE) using cross-validation with 10 folds, along with its 
    standard deviation.

    """
    # Compute predictions using cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=10, scoring='neg_mean_absolute_error')
    
    # Calculate the average MAE across all folds for cross-validation
    average_cv_mae = np.mean(-cv_scores)  # Take the negative since cross_val_score returns negative MAE

    # Calculate the average std
    std_cv_mae = np.std(cv_scores)

    # Fit the model on the training data
    model.fit(X_train, y_train)

    # Compute predictions on the training set
    y_pred = model.predict(X_train)

    # Calculate the Mean Squared Error (MSE) on the training set
    mae = mean_absolute_error(y_train, y_pred)
    
    print(f"{model_name} - Mean Absolute Error (MSE) on training set: {mae:.2f}")
    print(f"{model_name} - Mean Absolute Error (MAE) using cross-validation: {average_cv_mae:.2f} ± {std_cv_mae:.2f}")
    print()



if __name__ == "__main__":
    hourly_dataset  = fetch_ucirepo(id=275)
    features = hourly_dataset.data.features
    features = features.drop(['dteday'], axis=1) 
    target = hourly_dataset.data.targets
    target = target['cnt'].values
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

    
    # Initialize models
    dummy_regressor = DummyRegressor(strategy='mean')
    linear_regressor = LinearRegression()
    decision_tree_regressor = DecisionTreeRegressor()
    random_forest_regressor = RandomForestRegressor()
    gradient_boosting_regressor = GradientBoostingRegressor()
    
    
    # Train and evaluate models
    train_and_evaluate_model(dummy_regressor, "Dummy Regressor", X_train, y_train)
    train_and_evaluate_model(linear_regressor, "Linear Regressor", X_train, y_train)
    train_and_evaluate_model(decision_tree_regressor, "Decision Tree Regressor", X_train, y_train)
    train_and_evaluate_model(random_forest_regressor, "Random Forest Regressor", X_train, y_train)
    train_and_evaluate_model(gradient_boosting_regressor, "Gradient Boosting Regressor", X_train, y_train)
    