import pandas as pd
from sklearn.preprocessing import LabelEncoder
from typing import Tuple
from rules_engine import prepare_data

def load_data(file_path: str) -> pd.DataFrame:
    """
    Load data from a CSV file.

    Parameters
    ----------
    file_path : str
        Path to the CSV file.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the data.
    """
    return pd.read_csv(file_path)

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess data by converting categorical variables to numeric.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with categorical variables converted to numeric.
    """
    le = LabelEncoder()
    for column in df.columns:
        if df[column].dtype == 'object':
            df[column] = le.fit_transform(df[column])
    return df

def apply_smote(X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Apply SMOTE to balance the data.

    Parameters
    ----------
    X : pd.DataFrame
        Input features.
    y : pd.Series
        Target variable.
    
    Returns
    -------
    pd.DataFrame
        Resampled features.
    pd.Series
        Resampled target variable.
    """
    smote = SMOTE(random_state=0)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    return X_resampled, y_resampled

def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicate rows from the DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with duplicate rows removed.

    """
    return df.drop_duplicates()

def drop_columns(df, columns):
    """
    Drop specified columns from the DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    columns : List[str]
        List of column names to drop.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with specified columns removed.
    """
    return df.drop(columns, axis=1)

def process_data(input_file: str, output_file: str) -> None:
    """
    Load, preprocess, and save data.

    Parameters
    ----------
    input_file : str
        Path to the input CSV file.
    output_file : str
        Path to the output CSV file.
    """
    df = load_data(input_file)
    df = preprocess_data(df)
    
    # Apply SMOTE if target column is present
    if 'target' in df.columns:
        X = df.drop('target', axis=1)
        y = df['target']
        X_resampled, y_resampled = apply_smote(X, y)
        df_resampled = pd.concat([X_resampled, y_resampled], axis=1)
    else:
        df_resampled = df

    df_resampled = prepare_data(df_resampled)
    # df_resampled = drop_columns(df_resampled, ['AppID', 'IssueDate'])
    
    df_resampled.to_csv(output_file, index=False)

process_data('../data/raw/Dummy-Data.csv', '../data/processed/Dummy-Data.csv')
process_data('../data/raw/data-1000.csv', '../data/processed/data-1000.csv')
