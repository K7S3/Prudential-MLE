import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, KFold


def decision_tree_regression(
    data: pd.DataFrame, target: str, k: int = 5
) -> DecisionTreeRegressor:
    """
    Performs decision tree regression with k-fold cross-validation.

    Args:
        data (pd.DataFrame): The dataset.
        target (str): The target variable for regression.
        k (int): Number of folds for k-fold cross-validation.
    """
    X = data.drop(columns=[target])
    y = data[target]

    model = DecisionTreeRegressor(random_state=1)

    kf = KFold(n_splits=k, shuffle=True, random_state=1)

    mse_scores = -cross_val_score(model, X, y, cv=kf, scoring="neg_mean_squared_error")
    rmse_scores = np.sqrt(mse_scores)
    r2_scores = cross_val_score(model, X, y, cv=kf, scoring="r2")

    print(f"Average RMSE: {np.mean(rmse_scores)}")
    print(f"Average R2: {np.mean(r2_scores)}")

    model.fit(X, y)

    return model


def inference(
    model: DecisionTreeRegressor, test_data: pd.DataFrame, target: str
) -> None:
    """
    Make predictions on the test data and evaluate them.

    Args:
        model (DecisionTreeRegressor): The trained model.
        test_data (pd.DataFrame): The test data.
        target (str): The target variable for regression.
    """
    X_test = test_data.drop(columns=[target])
    y_test = test_data[target]

    y_pred = model.predict(X_test)

    print(f"Test RMSE: {np.sqrt(mean_squared_error(y_test, y_pred))}")
    print(f"Test R2: {r2_score(y_test, y_pred)}")

    return y_pred


def main():
    # Load the training data
    train_data = pd.read_csv("../data/processed/data-10000.csv")
    train_data = train_data.drop(columns=["AppID", "IssueDate", "Quote", "Reason"])
    target_variable = "BMI"

    # Train the model
    model = decision_tree_regression(train_data, target_variable)

    # Load the test data
    test_data = pd.read_csv("../data/processed/data-1000.csv")
    test_data = test_data.drop(columns=["AppID", "IssueDate", "Quote", "Reason"])

    # Conduct inference
    y_pred = inference(model, test_data, target_variable)

    # Save predictions to CSV
    pd.DataFrame(y_pred, columns=["Predicted"]).to_csv(
        "../data/predicted/dt-data-1000.csv", index=False
    )


if __name__ == "__main__":
    main()
