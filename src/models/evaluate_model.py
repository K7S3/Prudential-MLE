import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix

def calculate_disparate_impact(df: pd.DataFrame, protected_attribute: str, positive_outcome: str) -> float:
    """
    Calculate Disparate Impact.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the data.
    protected_attribute : str
        Name of the protected attribute column.
    positive_outcome : str
        Name of the positive outcome column.

    Returns
    -------
    float
        Disparate Impact.
    """
    group_counts = df.groupby(protected_attribute)[positive_outcome].value_counts(normalize=True).unstack()
    disparate_impact = group_counts.loc[1, True] / group_counts.loc[0, True]
    return disparate_impact

def calculate_statistical_parity_difference(df: pd.DataFrame, protected_attribute: str, positive_outcome: str) -> float:
    """
    Calculate Statistical Parity Difference.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the data.
    protected_attribute : str
        Name of the protected attribute column.
    positive_outcome : str
        Name of the positive outcome column.

    Returns
    -------
    float
        Statistical Parity Difference.
    """
    group_counts = df.groupby(protected_attribute)[positive_outcome].value_counts(normalize=True).unstack()
    statistical_parity_difference = group_counts.loc[1, True] - group_counts.loc[0, True]
    return statistical_parity_difference

def calculate_equalized_odds(y_true: np.ndarray, y_pred: np.ndarray, protected_attribute: np.ndarray) -> dict:
    """
    Calculate Equalized Odds.

    Parameters
    ----------
    y_true : np.ndarray
        True labels.
    y_pred : np.ndarray
        Predicted labels.
    protected_attribute : np.ndarray
        Protected attribute array.

    Returns
    -------
    dict
        Dictionary with TPR and FPR for each group.
    """
    cm_0 = confusion_matrix(y_true[protected_attribute == 0], y_pred[protected_attribute == 0])
    cm_1 = confusion_matrix(y_true[protected_attribute == 1], y_pred[protected_attribute == 1])

    tpr_0 = cm_0[1, 1] / (cm_0[1, 0] + cm_0[1, 1])
    fpr_0 = cm_0[0, 1] / (cm_0[0, 0] + cm_0[0, 1])

    tpr_1 = cm_1[1, 1] / (cm_1[1, 0] + cm_1[1, 1])
    fpr_1 = cm_1[0, 1] / (cm_1[0, 0] + cm_1[0, 1])

    return {
        'TPR_0': tpr_0,
        'FPR_0': fpr_0,
        'TPR_1': tpr_1,
        'FPR_1': fpr_1
    }

def calculate_predictive_parity(y_true: np.ndarray, y_pred: np.ndarray, protected_attribute: np.ndarray) -> dict:
    """
    Calculate Predictive Parity.

    Parameters
    ----------
    y_true : np.ndarray
        True labels.
    y_pred : np.ndarray
        Predicted labels.
    protected_attribute : np.ndarray
        Protected attribute array.

    Returns
    -------
    dict
        Predictive Parity for each group.
    """
    ppv_0 = np.sum((y_true == 1) & (y_pred == 1) & (protected_attribute == 0)) / np.sum((y_pred == 1) & (protected_attribute == 0))
    ppv_1 = np.sum((y_true == 1) & (y_pred == 1) & (protected_attribute == 1)) / np.sum((y_pred == 1) & (protected_attribute == 1))

    return {
        'PPV_0': ppv_0,
        'PPV_1': ppv_1
    }
def evaluate_fairness(X: pd.DataFrame, y: pd.Series, protected_attribute: pd.Series, model) -> dict:
    """
    Evaluate fairness of the model.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Target array.
    protected_attribute : pd.Series
        Protected attribute array.
    model : object
        Trained model.

    Returns
    -------
    dict
        Fairness metrics.
    """
    y_pred = model.predict(X)

    df = pd.DataFrame({
        'predicted_outcome': y_pred,
        'protected_attribute': protected_attribute
    })

    metrics = {}
    metrics['Disparate Impact'] = calculate_disparate_impact(df, 'protected_attribute', 'predicted_outcome')
    metrics['Statistical Parity Difference'] = calculate_statistical_parity_difference(df, 'protected_attribute', 'predicted_outcome')
    metrics['Equalized Odds'] = calculate_equalized_odds(y.values, y_pred, protected_attribute.values)
    metrics['Predictive Parity'] = calculate_predictive_parity(y.values, y_pred, protected_attribute.values)

    return metrics

# Example usage:
# df = pd.read_csv('your_data.csv')
# model = train_your_model(df)
# fairness_metrics = evaluate_fairness(df, 'protected_attribute', df['target'], model)
# print(fairness_metrics)