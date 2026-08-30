from src.retraining.retrain import retrain_model
from src.retraining.validation import validate_model


def auto_retrain(
    arcs_score,
    model,
    X_train,
    y_train,
    X_test,
    y_test,
    threshold=0.70
):
    """
    Automatically retrain the model when ARCS exceeds
    the retraining threshold.
    """

    print("========== AUTO RETRAINING ==========")
    print("ARCS Score:", round(float(arcs_score), 4))
    print("Retraining Threshold:", threshold)

    # Check ARCS decision
    if arcs_score < threshold:
        print("🟢 ARCS is below retraining threshold.")
        print("No retraining required.")

        return {
            "retraining_required": False,
            "model": model,
            "validation": None
        }

    # Retraining required
    print("🔴 ARCS is above retraining threshold.")
    print("Retraining required.")

    # Retrain model
    new_model = retrain_model(
        model,
        X_train,
        y_train
    )

    # Validate new model
    validation_results = validate_model(
        new_model,
        X_test,
        y_test
    )

    print("========== AUTO RETRAINING COMPLETE ==========")

    return {
        "retraining_required": True,
        "model": new_model,
        "validation": validation_results
    }