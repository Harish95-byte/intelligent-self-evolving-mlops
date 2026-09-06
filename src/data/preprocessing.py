import pandas as pd
from sklearn.model_selection import train_test_split


def load_data(csv_path):
    """
    Load the Telco Customer Churn dataset.
    """
    return pd.read_csv(csv_path)


def preprocess_data(data):
    """
    Clean and encode the Telco Customer Churn dataset.
    """

    df = data.copy()

    # Convert TotalCharges to numeric
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"],
        errors="coerce"
    )

    # Remove rows with invalid TotalCharges
    df = df.dropna()

    # Convert target to binary
    df["Churn"] = df["Churn"].map({
        "Yes": 1,
        "No": 0
    })

    # Remove customer identifier
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    # One-hot encode categorical variables
    df = pd.get_dummies(
        df,
        drop_first=True
    )

    # Separate features and target
    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    return X, y


def split_data(X, y, test_size=0.20, random_state=42):
    """
    Split data into training and testing sets.
    """

    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )
