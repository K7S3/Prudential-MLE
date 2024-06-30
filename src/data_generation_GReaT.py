import os
# Disable MPS
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import pandas as pd
import numpy as np
import torch
import argparse

from be_great import GReaT

def height_to_inches(height: int) -> int:
    """Converts height from ft and inches format to total inches.
    
    Args:
        height (int): Height in ft and inches format, e.g., '507'.
        
    Returns:
        int: Total height in inches.
    """
    feet = height // 100
    inches = height % 100
    return feet * 12 + inches

def inches_to_height(inches: int) -> str:
    """Converts total inches to height in ft and inches format.
    
    Args:
        inches (int): Total height in inches.
        
    Returns:
        str: Height in ft and inches format, e.g., '507'.
    """
    feet = inches // 12
    remaining_inches = inches % 12
    return f"{feet}{remaining_inches:02d}"

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocesses the data for model training.
    
    Args:
        df (pd.DataFrame): Original dataset.
        
    Returns:
        pd.DataFrame: Preprocessed dataset.
    """
    df['Ht'] = df['Ht'].apply(height_to_inches)
    df['Ins_Gender'] = df['Ins_Gender'].apply(lambda x: 0 if x == 'Male' else 1)
    df['IssueDate'] = pd.to_datetime(df['IssueDate']).map(pd.Timestamp.to_julian_date)
    return df

def postprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Postprocesses the generated data to the original format.
    
    Args:
        df (pd.DataFrame): Generated dataset.
        
    Returns:
        pd.DataFrame: Postprocessed dataset.
    """
    df['Ins_Gender'] = df['Ins_Gender'].apply(lambda x: 'Male' if x < 0.5 else 'Female')
    df['Ht'] = df['Ht'].apply(lambda x: inches_to_height(int(x)))
    df['IssueDate'] = pd.to_datetime(df['IssueDate'], origin='julian', unit='D')
    return df

def generate_data_GReaT(num_samples: int, device: str, use_fp16: bool) -> pd.DataFrame:
    """
    Generate synthetic data using the GReaT library.

    Parameters
    ----------
    num_samples : int
        Number of samples to generate.
    device : str
        Device to use ('cuda' or 'cpu').
    use_fp16 : bool
        Whether to use fp16 precision.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with synthetic data.
    """
    original_data = pd.read_csv('../data/raw/Dummy-Data.csv')
    data = preprocess_data(original_data)

    # Initialize the GReaT model
    great = GReaT(llm='gpt2', batch_size=4, epochs=50, fp16=use_fp16)
    
    # Ensure the model is moved to the correct device
    great.model.to(device)
    print(f"Model is on device: {next(great.model.parameters()).device}")

    # Fit the model on the data
    great.fit(data)  # Ensure data is passed as a DataFrame or Numpy array

    # Generate synthetic data
    synthetic_data = great.sample(n_samples=num_samples, device=device)

    # Postprocess the generated data
    synthetic_data = postprocess_data(synthetic_data)
    
    return synthetic_data

def main(num_samples: int) -> None:
    """Main function to generate and save synthetic data.
    
    Args:
        num_samples (int): Number of samples to generate.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    use_fp16 = torch.cuda.is_available()
    
    synthetic_data = generate_data_GReaT(num_samples, device, use_fp16)
    output_path = f'../data/raw/llm_data_{num_samples}.csv'
    synthetic_data.to_csv(output_path, index=False)
    print(f"Synthetic data generation complete. Data saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic data using GPT-2.")
    parser.add_argument('--samples', default=1000, type=int, help='Number of samples to generate')
    args = parser.parse_args()
    
    main(args.samples)
