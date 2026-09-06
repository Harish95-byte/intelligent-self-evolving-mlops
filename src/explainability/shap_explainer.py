import shap
import pandas as pd
import numpy as np


def explain_model(model, X, feature_names=None, max_display=10):
    """
    Generate SHAP explanations for a tree-based classification model.
    Works with different SHAP output formats.
    """

    if feature_names is None:
        if hasattr(X, "columns"):
            feature_names = list(X.columns)
        else:
            feature_names = [f"feature_{i}" for i in range(X.shape[1])]

    explainer = shap.TreeExplainer(model)

    shap_values = explainer.shap_values(X)

    # Handle different SHAP output formats
    if isinstance(shap_values, list):
        values = np.asarray(shap_values[-1])
    else:
        values = np.asarray(shap_values)

        # Newer SHAP versions may return:
        # (samples, features, classes)
        if values.ndim == 3:
            values = values[:, :, -1]

    # Ensure 2-dimensional SHAP values
    if values.ndim != 2:
        raise ValueError(
            f"Unexpected SHAP output shape: {values.shape}"
        )

    importance = np.abs(values).mean(axis=0)

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": importance
    }).sort_values(
        "mean_abs_shap",
        ascending=False
    ).reset_index(drop=True)

    print("========== SHAP / XAI ANALYSIS ==========")
    print(importance_df.head(max_display).to_string(index=False))

    return {
        "shap_values": values,
        "feature_importance": importance_df
    }
