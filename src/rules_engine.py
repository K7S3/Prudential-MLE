from typing import Tuple
import pandas as pd


def height_to_meters(height: int) -> float:
    """
    Convert height from feet and inches to meters.

    Parameters
    ----------
    height : int
        Height in feet and inches.

    Returns
    -------
    float
        Height in meters.
    """
    feet = height // 100
    inches = height % 100
    total_inches = feet * 12 + inches
    meters = total_inches * 0.0254
    return meters


def weight_to_kg(weight: int) -> float:
    """
    Convert weight from pounds to kilograms.

    Parameters
    ----------
    weight : int
        Weight in pounds.

    Returns
    -------
    float
        Weight in kilograms.
    """
    kg = weight * 0.453592
    return kg


def calculate_bmi(height: int, weight: int) -> float:
    """
    Calculate BMI from height and weight.

    Parameters
    ----------
    height : int
        Height in feet and inches.
    weight : int
        Weight in pounds.

    Returns
    -------
    float
        BMI value.
    """
    height_m = height_to_meters(height)
    weight_kg = weight_to_kg(weight)
    bmi = weight_kg / (height_m**2)
    return bmi


def determine_quote(age: int, bmi: float, gender: str) -> Tuple[int, str]:
    """
    Determine insurance quote based on age, BMI, and gender.

    Parameters
    ----------
    age : int
        Age of the person.
    bmi : float
        BMI value of the person.
    gender : str
        Gender of the person.

    Returns
    -------
    Tuple[int, str]
        Insurance quote and reason for the quote based on business rules.
    """
    if (age >= 18 and age <= 39) and (bmi < 17.49 or bmi > 38.5):
        quote = 750
        reason = "Age is between 18 to 39 and BMI is either less than 17.49 or greater than 38.5"
    elif (age >= 40 and age <= 59) and (bmi < 18.49 or bmi > 38.5):
        quote = 1000
        reason = "Age is between 40 to 59 and BMI is either less than 18.49 or greater than 38.5"
    elif age >= 60 and (bmi < 18.49 or bmi > 45.5):
        quote = 2000
        reason = "Age is greater than 60 and BMI is either less than 18.49 or greater than 38.5"
    else:
        quote = 500
        reason = "BMI is in right range"

    if gender == "Female":
        quote *= 0.9

    return quote, reason


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare data by calculating BMI and determining insurance quote.

    Parameters
    ----------
    df : pd.DataFrame
        Input pandas dataframe containing height
        and weight in feet and inches and pounds respectively.

    Returns
    -------
    pd.DataFrame
        Output pandas dataframe with BMI, insurance quote, and reason for the quote.
    """
    df["BMI"] = df.apply(lambda row: calculate_bmi(row["Ht"], row["Wt"]), axis=1)
    df["Quote"], df["Reason"] = zip(
        *df.apply(
            lambda row: determine_quote(row["Ins_Age"], row["BMI"], row["Ins_Gender"]),
            axis=1,
        )
    )
    return df
