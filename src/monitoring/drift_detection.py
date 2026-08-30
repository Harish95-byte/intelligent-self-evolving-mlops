import pandas as pd
from scipy.stats import ks_2samp


def detect_drift(reference_data, production_data, numerical_columns):

    drift_results = {}

    for column in numerical_columns:

        reference_values = reference_data[column].dropna()
        production_values = production_data[column].dropna()

        statistic, p_value = ks_2samp(
            reference_values,
            production_values
        )

        drift_results[column] = {
            "statistic": round(statistic, 4),
            "p_value": round(p_value, 4),
            "drift_detected": p_value < 0.05
        }

    return drift_results
