def simulate_customer(
    tenure,
    monthly_charges,
    contract,
    churn_probability
):
    """
    Simulate a customer intervention scenario.
    """

    tenure = float(tenure)
    monthly_charges = float(monthly_charges)
    churn_probability = float(churn_probability)

    current_state = {
        "tenure": tenure,
        "monthly_charges": monthly_charges,
        "contract": contract,
        "churn_probability": round(churn_probability, 4)
    }

    # Simulated intervention:
    # assume a retention intervention reduces predicted churn risk by 10 percentage points
    simulated_probability = max(
        0.0,
        churn_probability - 0.10
    )

    intervention_state = {
        "tenure": tenure,
        "monthly_charges": monthly_charges,
        "contract": contract,
        "churn_probability": round(
            simulated_probability, 4
        )
    }

    risk_reduction = (
        churn_probability -
        simulated_probability
    )

    return {
        "current_state": current_state,
        "intervention_state": intervention_state,
        "risk_reduction": round(
            risk_reduction, 4
        )
    }
