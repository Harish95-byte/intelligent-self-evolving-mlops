def auto_retrain(arcs_score, threshold=0.70):
    """
    Decide whether the model needs retraining
    based on the ARCS score.
    """

    if arcs_score < threshold:
        print("ARCS is below retraining threshold.")
        print("No retraining required.")
        return False

    print("⚠️ ARCS is high.")
    print("Retraining required.")

    return True