import numpy as np


def predict_customer(model, X):
    """
    Generate churn predictions and probabilities.
    """

    predictions = model.predict(X)

    # Probability of churn
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X)[:, 1]
    else:
        probabilities = np.zeros(len(predictions))

    return {
        "predictions": predictions,
        "churn_probability": probabilities
    }


def predict_single_customer(model, customer_data):
    """
    Predict churn for a single processed customer.
    """

    result = predict_customer(
        model,
        customer_data
    )

    return {
        "prediction": int(result["predictions"][0]),
        "churn_probability": round(
            float(result["churn_probability"][0]),
            4
        )
    }
