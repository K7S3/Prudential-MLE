import numpy as np
import pandas as pd


def generate_data_random(num_samples: int) -> pd.DataFrame:
    """
    Generate synthetic data with random values.

    Parameters
    ----------
    num_samples : int
        Number of samples to generate.

    Returns
    -------
    pd.DataFrame
        DataFrame with synthetic data.
    """

    feet = np.random.randint(1, 9, num_samples)
    inches = np.random.randint(0, 12, num_samples)
    height = feet * 100 + inches

    weight = np.random.randint(50, 500, num_samples)
    age = np.random.randint(18, 100, num_samples)

    gender = np.random.choice(["Male", "Female"], num_samples)

    issue_date = pd.date_range(start="1/1/2010", end="12/31/2023").to_series()
    issue_date = issue_date.sample(num_samples, replace=True).dt.date

    app_id = np.random.randint(10000, 1000000, num_samples)

    df = pd.DataFrame(
        {
            "AppID": app_id,
            "Ins_Age": age,
            "Ins_Gender": gender,
            "Ht": height,
            "Wt": weight,
            "IssueDate": issue_date,
        }
    )

    return df


num_samples = 10000

df = generate_data_random(num_samples)

directory = "../data/raw/"

df.to_csv(f"{directory}data-{num_samples}.csv", index=False)
