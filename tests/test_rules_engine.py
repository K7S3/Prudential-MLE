import pytest
import pandas as pd
from hypothesis import given, strategies as st
from src import data_preparation


@given(st.integers(min_value=400, max_value=700), st.integers(min_value=100, max_value=300))
def test_calculate_bmi(height, weight):
    bmi = data_preparation.calculate_bmi(height, weight)
    assert isinstance(bmi, float)
    assert 0 <= bmi <= 100  

@given(st.integers(min_value=18, max_value=70), st.floats(min_value=15.0, max_value=45.0), st.sampled_from(['Male', 'Female']))
def test_determine_quote(age, bmi, gender):
    quote, reason = data_preparation.determine_quote(age, bmi, gender)
    assert isinstance(quote, (int, float))
    assert isinstance(reason, str)
    assert quote >= 0  

def test_prepare_data():
    df = pd.DataFrame({
        'AppID': [1, 2, 3],
        'Ht': [507, 600, 511],
        'Wt': [150, 200, 180],
        'Ins_Age': [30, 50, 60],
        'Ins_Gender': ['Male', 'Female', 'Male']
    })
    df_processed = data_preparation.prepare_data(df)
    assert 'BMI' in df_processed.columns
    assert 'Quote' in df_processed.columns
    assert 'Reason' in df_processed.columns
    assert df_processed['BMI'].isnull().sum() == 0  
    assert df_processed['Quote'].isnull().sum() == 0  
    assert df_processed['Reason'].isnull().sum() == 0  