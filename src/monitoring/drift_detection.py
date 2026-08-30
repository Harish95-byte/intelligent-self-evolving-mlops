import pandas as pd
import numpy as np


def detect_drift(reference_data, production_data, numerical_columns):
    """
    Detect distribution drift between reference and production data.
    """

    drift_results = {}

    for column in numerical_columns:

        reference_mean = reference_data[column].mean()
        production_mean = production_data[column].mean()

        if reference_mean == 0:
            drift_score = 0
        else:
            drift_score = abs(
                production_mean - reference_mean
            ) / abs(reference_mean)

        drift_results[column] = round(drift_score, 4)

    overall_drift = np.mean(list(drift_results.values()))

    return {
        "feature_drift": drift_results,
        "overall_drift": round(overall_drift, 4)
    }