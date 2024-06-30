import pandas as pd
from ctgan import CTGAN
from sklearn.preprocessing import LabelEncoder

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocesses the data for CTGAN training.

    Args:
        df (pd.DataFrame): Original dataset.

    Returns:
        pd.DataFrame: Preprocessed dataset.
    """
    le_gender = LabelEncoder()
    df['Ins_Gender'] = le_gender.fit_transform(df['Ins_Gender'])
    
    df['IssueDate'] = pd.to_datetime(df['IssueDate'], errors='coerce')
    df = df.dropna(subset=['IssueDate'])
    df['IssueDate'] = df['IssueDate'].map(pd.Timestamp.toordinal)
    
    # Remove columns with no variance
    df = df.loc[:, df.apply(pd.Series.nunique) > 1]

    return df


def postprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Postprocesses the generated data to the original format.

    Args:
        df (pd.DataFrame): Generated dataset.

    Returns:
        pd.DataFrame: Postprocessed dataset.
    """
    le_gender = LabelEncoder()
    le_gender.fit(['Male', 'Female'])
    df['Ins_Gender'] = le_gender.inverse_transform(df['Ins_Gender'])

    df['IssueDate'] = pd.to_datetime(df['IssueDate'], origin='julian', unit='D')

    return df

def generate_synthetic_data(data: pd.DataFrame, discrete_columns: list, num_samples: int, epochs: int = 300, batch_size: int = 500) -> pd.DataFrame:
    """
    Generates synthetic data using the CTGAN model.

    Args:
        data (pd.DataFrame): The input data.
        discrete_columns (list): List of discrete columns in the data.
        num_samples (int): Number of samples to generate.
        epochs (int): Number of training epochs.
        batch_size (int): Batch size for training.

    Returns:
        pd.DataFrame: DataFrame containing the generated data.
    """
    model = CTGAN(epochs=epochs, batch_size=batch_size)
    model.fit(data, discrete_columns)
    synthetic_data = model.sample(num_samples)
    return synthetic_data

def main():
    """
    The main function to generate synthetic tabular data and save it to a CSV file.
    """
    original_data = pd.read_csv('../data/raw/Dummy-Data.csv')
    data = preprocess_data(original_data)

    discrete_columns = ['Ins_Gender']

    num_samples = 100
    synthetic_data = generate_synthetic_data(data, discrete_columns, num_samples)

    synthetic_data = postprocess_data(synthetic_data)

    output_path = f'generated_data_{num_samples}.csv'
    synthetic_data.to_csv(output_path, index=False)
    print(f"Synthetic data generation complete. Data saved to {output_path}")

if __name__ == "__main__":
    main()
