from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


def evaluate_model(model, X_test, y_test):
    """
    Evaluate a trained classification model.
    """

    y_pred = model.predict(X_test)

    results = {
        "accuracy": round(
            float(accuracy_score(y_test, y_pred)), 4
        ),
        "precision": round(
            float(precision_score(
                y_test,
                y_pred,
                zero_division=0
            )), 4
        ),
        "recall": round(
            float(recall_score(
                y_test,
                y_pred,
                zero_division=0
            )), 4
        ),
        "f1": round(
            float(f1_score(
                y_test,
                y_pred,
                zero_division=0
            )), 4
        )
    }

    return results
