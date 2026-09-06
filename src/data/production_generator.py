import pandas as pd
import numpy as np


def generate_production_data(
    reference_data,
    sample_size=1000,
    random_state=42
):
    """
    Generate simulated production data based on
    the reference dataset.
    """

    rng = np.random.default_rng(random_state)

    # Sample rows from reference data
    production_data = reference_data.sample(
        n=min(sample_size, len(reference_data)),
        replace=True,
        random_state=random_state
    ).copy()

    # Add small numerical variation to simulate
    # real-world production changes
    numeric_columns = production_data.select_dtypes(
        include=np.number
    ).columns

    for column in numeric_columns:
        values = production_data[column].astype(float)

        noise = rng.normal(
            loc=0.0,
            scale=max(values.std() * 0.05, 0.001),
            size=len(production_data)
        )

        production_data[column] = values + noise

    return production_data.reset_index(drop=True)
