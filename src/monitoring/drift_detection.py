import pandas as pd
from scipy.stats import ks_2samp


def detect_drift(reference_data, production_data, numerical_columns):

    drift_results = {}

    for column in numerical_columns:

        # Convert values to numeric
        reference_values = pd.to_numeric(
            reference_data[column],
            errors="coerce"
        ).dropna()

        production_values = pd.to_numeric(
            production_data[column],
            errors="coerce"
        ).dropna()

        # Skip if there is not enough data
        if len(reference_values) == 0 or len(production_values) == 0:
            drift_results[column] = {
                "statistic": 0.0,
                "p_value": 1.0,
                "drift_detected": False
            }
            continue

        statistic, p_value = ks_2samp(
            reference_values,
            production_values
        )

        drift_results[column] = {
            "statistic": round(float(statistic), 4),
            "p_value": round(float(p_value), 4),
            "drift_detected": bool(p_value < 0.05)
        }

    return drift_results
