from scipy.stats import ks_2samp


def detect_drift(reference_data, production_data, numerical_columns):
    """
    Detect feature drift using the Kolmogorov-Smirnov test.
    """

    drift_results = {}

    for column in numerical_columns:

        statistic, p_value = ks_2samp(
            reference_data[column],
            production_data[column]
        )

        drift_detected = p_value < 0.05

        drift_results[column] = {
            "ks_statistic": statistic,
            "p_value": p_value,
            "drift_detected": drift_detected
        }

    return drift_results
