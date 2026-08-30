from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


def calculate_performance(y_true, y_pred):
    """
    Calculate classification performance metrics.
    """

    accuracy = accuracy_score(y_true, y_pred)

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0
    )

    return {
        "accuracy": round(float(accuracy), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1": round(float(f1), 4)
    }


def calculate_performance_risk(f1_score_value, target_f1=0.60):
    """
    Calculate performance risk based on F1 score.
    """

    risk = max(
        0,
        target_f1 - f1_score_value
    )

    return round(float(risk), 4)