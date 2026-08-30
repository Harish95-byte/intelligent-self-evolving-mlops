from sklearn.base import clone


def retrain_model(model, X_train, y_train):
    """
    Retrain a copy of the existing model using new training data.
    """

    new_model = clone(model)

    new_model.fit(X_train, y_train)

    print("✅ Model retraining completed successfully.")

    return new_model