def make_decision(
    churn_probability,
    causal_effect,
    risk_reduction,
    arcs_score
):
    """
    Convert model, causal, digital-twin and ARCS
    results into an actionable decision.
    """

    churn_probability = float(churn_probability)
    causal_effect = float(causal_effect)
    risk_reduction = float(risk_reduction)
    arcs_score = float(arcs_score)

    # Customer risk decision
    if churn_probability >= 0.70:
        customer_action = "HIGH RISK - Retention Action Required"
    elif churn_probability >= 0.40:
        customer_action = "MEDIUM RISK - Monitor Customer"
    else:
        customer_action = "LOW RISK - No Immediate Action"

    # Intervention decision
    if risk_reduction > 0:
        intervention = "Intervention Recommended"
    else:
        intervention = "No Intervention Required"

    # Model lifecycle decision
    if arcs_score >= 0.70:
        model_action = "RETRAIN MODEL"
    elif arcs_score >= 0.30:
        model_action = "MONITOR MODEL"
    else:
        model_action = "CONTINUE CURRENT MODEL"

    return {
        "customer_action": customer_action,
        "intervention": intervention,
        "model_action": model_action,
        "churn_probability": round(churn_probability, 4),
        "causal_effect": round(causal_effect, 4),
        "risk_reduction": round(risk_reduction, 4),
        "arcs_score": round(arcs_score, 4)
    }
