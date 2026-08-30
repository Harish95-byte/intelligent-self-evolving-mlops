from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


def validate_model(model, X_test, y_test):
    """
    Validate the retrained model.
    """

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    results = {
        "accuracy": round(float(accuracy), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1": round(float(f1), 4)
    }

    print("========== RETRAINED MODEL VALIDATION ==========")
    print("Accuracy :", results["accuracy"])
    print("Precision:", results["precision"])
    print("Recall   :", results["recall"])
    print("F1-Score :", results["f1"])

    return results