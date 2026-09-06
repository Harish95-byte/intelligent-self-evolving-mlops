def analyze_causal_impact(
    baseline,
    intervention,
    metric="churn_probability"
):
    """
    Estimate the impact of an intervention by comparing
    baseline and intervention outcomes.

    Parameters
    ----------
    baseline : float
        Outcome before intervention.

    intervention : float
        Outcome after intervention.

    metric : str
        Name of the metric being analyzed.

    Returns
    -------
    dict
        Causal impact summary.
    """

    baseline = float(baseline)
    intervention = float(intervention)

    absolute_change = intervention - baseline

    if baseline != 0:
        percentage_change = (
            absolute_change / abs(baseline)
        ) * 100
    else:
        percentage_change = 0.0

    if absolute_change < 0:
        effect = "IMPROVEMENT"
    elif absolute_change > 0:
        effect = "DEGRADATION"
    else:
        effect = "NO_CHANGE"

    return {
        "metric": metric,
        "baseline": round(baseline, 4),
        "intervention": round(intervention, 4),
        "absolute_change": round(absolute_change, 4),
        "percentage_change": round(percentage_change, 2),
        "effect": effect
    }