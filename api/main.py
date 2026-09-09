from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response

import os
import json
import time
import joblib
import pandas as pd
import numpy as np

from sklearn.base import clone
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

from src.monitoring.drift_detection import detect_drift
from src.monitoring.performance_monitor import calculate_performance_risk


# ============================================================
# APP
# ============================================================

app = FastAPI(
    title="Intelligent Self-Evolving MLOps",
    description="Production-style self-evolving ML platform",
    version="2.0",
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_V1_PATH = os.path.join(BASE_DIR, "model_v1.pkl")
PREPROCESSOR_PATH = os.path.join(BASE_DIR, "preprocessor_v1.pkl")
METADATA_PATH = os.path.join(BASE_DIR, "model_v1_metadata.json")

DATASET_PATH = os.path.join(
    BASE_DIR,
    "WA_Fn-UseC_-Telco-Customer-Churn.csv"
)

MODEL_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(MODEL_DIR, exist_ok=True)

ACTIVE_MODEL_FILE = os.path.join(
    MODEL_DIR,
    "production.json"
)


# ============================================================
# LOAD PREPROCESSOR
# ============================================================

preprocessor = joblib.load(PREPROCESSOR_PATH)


# ============================================================
# ACTIVE MODEL MANAGEMENT
# ============================================================

def initialize_production_model():

    v1_destination = os.path.join(
        MODEL_DIR,
        "model_v1.pkl"
    )

    if not os.path.exists(v1_destination):
        import shutil
        shutil.copy2(
            MODEL_V1_PATH,
            v1_destination
        )

    if not os.path.exists(ACTIVE_MODEL_FILE):

        with open(ACTIVE_MODEL_FILE, "w") as f:
            json.dump(
                {
                    "active_model": "model_v1.pkl",
                    "version": "v1"
                },
                f,
                indent=4
            )


initialize_production_model()


def get_active_model_info():

    with open(ACTIVE_MODEL_FILE, "r") as f:
        return json.load(f)


def get_active_model_path():

    info = get_active_model_info()

    return os.path.join(
        MODEL_DIR,
        info["active_model"]
    )


def load_active_model():

    return joblib.load(
        get_active_model_path()
    )


# ============================================================
# INITIAL MODEL
# ============================================================

model = load_active_model()


# ============================================================
# PRODUCTION BASELINE
# ============================================================

PRODUCTION_ACCURACY = 0.7896
PRODUCTION_PRECISION = 0.6258
PRODUCTION_RECALL = 0.5187
PRODUCTION_F1 = 0.5673


# ============================================================
# PROMETHEUS METRICS
# ============================================================

prediction_requests_total = Counter(
    "prediction_requests_total",
    "Total prediction requests"
)

predictions_total = Counter(
    "predictions_total",
    "Total successful predictions"
)

prediction_errors_total = Counter(
    "prediction_errors_total",
    "Total prediction errors"
)

prediction_latency = Histogram(
    "prediction_request_latency_seconds",
    "Prediction request latency"
)


model_accuracy = Gauge(
    "model_accuracy",
    "Current model accuracy"
)

model_precision = Gauge(
    "model_precision",
    "Current model precision"
)

model_recall = Gauge(
    "model_recall",
    "Current model recall"
)

model_f1 = Gauge(
    "model_f1_score",
    "Current model F1 score"
)

model_drift = Gauge(
    "model_drift_score",
    "Current model drift score"
)

model_risk = Gauge(
    "model_performance_risk",
    "Current model performance risk"
)

model_arcs = Gauge(
    "model_arcs_score",
    "Current ARCS score"
)

model_retraining = Gauge(
    "model_retraining_required",
    "Whether retraining is required"
)

model_version = Gauge(
    "model_version_number",
    "Current production model version"
)


# ============================================================
# INITIAL METRICS
# ============================================================

model_accuracy.set(PRODUCTION_ACCURACY)
model_precision.set(PRODUCTION_PRECISION)
model_recall.set(PRODUCTION_RECALL)
model_f1.set(PRODUCTION_F1)

model_drift.set(0.0304)
model_risk.set(0.0327)
model_arcs.set(0.688)
model_retraining.set(0)

model_version.set(1)


# ============================================================
# CUSTOMER INPUT
# ============================================================

class CustomerData(BaseModel):

    customerID: str
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
def health():

    info = get_active_model_info()

    return {
        "status": "healthy",
        "active_model": info["active_model"],
        "version": info["version"]
    }


# ============================================================
# MODEL STATUS
# ============================================================

@app.get("/model-status")
def model_status():

    info = get_active_model_info()

    return {
        "active_model": info["active_model"],
        "version": info["version"],
        "accuracy": model_accuracy._value.get(),
        "precision": model_precision._value.get(),
        "recall": model_recall._value.get(),
        "f1": model_f1._value.get(),
        "drift": model_drift._value.get(),
        "performance_risk": model_risk._value.get(),
        "arcs": model_arcs._value.get(),
        "retraining_required": bool(
            model_retraining._value.get()
        )
    }


# ============================================================
# PREDICTION
# ============================================================

@app.post("/predict")
def predict(data: CustomerData):

    global model

    start = time.time()

    prediction_requests_total.inc()

    try:

        active_info = get_active_model_info()

        # Reload active production model.
        # This is what allows V2/V3/etc. to become live.
        model = load_active_model()

        input_data = pd.DataFrame(
            [data.dict()]
        )

        input_data = input_data.drop(
            columns=["customerID"]
        )

        processed = preprocessor.transform(
            input_data
        )

        prediction = int(
            model.predict(processed)[0]
        )

        latency = time.time() - start

        prediction_latency.observe(
            latency
        )

        predictions_total.inc()

        return {
            "prediction": prediction,
            "model": active_info["active_model"],
            "version": active_info["version"],
            "status": "success",
            "latency_seconds": round(
                latency,
                4
            )
        }

    except Exception as e:

        prediction_errors_total.inc()

        return {
            "status": "error",
            "message": str(e)
        }


# ============================================================
# REAL RETRAINING ENGINE
# ============================================================

def train_candidate_model():

    print("\n================================")
    print("REAL MODEL RETRAINING STARTED")
    print("================================")

    # --------------------------------------------------------
    # Load real dataset
    # --------------------------------------------------------

    df = pd.read_csv(
        DATASET_PATH
    )
    # Clean TotalCharges values
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"],
        errors="coerce"
    )

    # Remove rows where TotalCharges could not be converted
    df = df.dropna(
        subset=["TotalCharges"]
    ).copy()

    # Convert target
    y = df["Churn"].map(
        {
            "Yes": 1,
            "No": 0
        }
    )

    X = df.drop(
        columns=["Churn", "customerID"]
    )

    # --------------------------------------------------------
    # Train/test split
    # --------------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    # --------------------------------------------------------
    # Transform data
    # --------------------------------------------------------

    X_train_processed = preprocessor.transform(
        X_train
    )

    X_test_processed = preprocessor.transform(
        X_test
    )

    # --------------------------------------------------------
    # Load current production model
    # --------------------------------------------------------

    current_model = load_active_model()

    # --------------------------------------------------------
    # Clone and train candidate
    # --------------------------------------------------------

    candidate_model = clone(
        current_model
    )

    candidate_model.fit(
        X_train_processed,
        y_train
    )

    # --------------------------------------------------------
    # Candidate validation
    # --------------------------------------------------------

    y_pred = candidate_model.predict(
        X_test_processed
    )

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

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

    validation = {

        "accuracy": round(
            float(accuracy),
            4
        ),

        "precision": round(
            float(precision),
            4
        ),

        "recall": round(
            float(recall),
            4
        ),

        "f1": round(
            float(f1),
            4
        )
    }

    print(
        "Candidate validation:",
        validation
    )

    return candidate_model, validation


# ============================================================
# SELF-EVOLUTION
# ============================================================

@app.post("/evolve")
def evolve():

    global model

    # --------------------------------------------------------
    # Controlled degradation trigger
    #
    # The trigger represents a production monitoring event.
    # The actual retraining below uses the REAL dataset.
    # --------------------------------------------------------

    simulated_drift = 0.90

    simulated_f1 = 0.05

    target_f1 = 0.60

    performance_risk = max(
        0,
        target_f1 - simulated_f1
    )

    arcs = (
        0.5 * simulated_drift
        +
        0.5 * performance_risk
    )

    arcs = min(
        max(arcs, 0),
        1
    )

    model_drift.set(
        simulated_drift
    )

    model_risk.set(
        performance_risk
    )

    model_arcs.set(
        arcs
    )

    # --------------------------------------------------------
    # Evolution decision
    # --------------------------------------------------------

    retraining_required = arcs >= 0.70

    model_retraining.set(
        1 if retraining_required else 0
    )

    if not retraining_required:

        return {
            "status": "no_retraining",
            "message": "Current model remains in production.",
            "arcs": round(arcs, 4)
        }

    # --------------------------------------------------------
    # ACTUAL RETRAINING
    # --------------------------------------------------------

    candidate_model, validation = train_candidate_model()

    # --------------------------------------------------------
    # Current production metrics
    # --------------------------------------------------------

    old_info = get_active_model_info()

    old_accuracy = float(
        model_accuracy._value.get()
    )

    old_f1 = float(
        model_f1._value.get()
    )

    new_accuracy = validation["accuracy"]

    new_f1 = validation["f1"]

    # --------------------------------------------------------
    # Candidate comparison
    # --------------------------------------------------------

    candidate_is_better = (
        new_f1 > old_f1
        or new_accuracy > old_accuracy
    )

    # --------------------------------------------------------
    # Generate new model version
    # --------------------------------------------------------

    current_version = int(
        old_info["version"].replace(
            "v",
            ""
        )
    )

    new_version_number = (
        current_version + 1
    )

    new_version = (
        f"v{new_version_number}"
    )

    new_model_filename = (
        f"model_{new_version}.pkl"
    )

    new_model_path = os.path.join(
        MODEL_DIR,
        new_model_filename
    )

    # --------------------------------------------------------
    # PROMOTION
    # --------------------------------------------------------

    if candidate_is_better:

        joblib.dump(
            candidate_model,
            new_model_path
        )

        with open(
            ACTIVE_MODEL_FILE,
            "w"
        ) as f:

            json.dump(
                {
                    "active_model":
                        new_model_filename,
                    "version":
                        new_version
                },
                f,
                indent=4
            )

        model = candidate_model

        model_accuracy.set(
            new_accuracy
        )

        model_precision.set(
            validation["precision"]
        )

        model_recall.set(
            validation["recall"]
        )

        model_f1.set(
            new_f1
        )

        model_version.set(
            new_version_number
        )

        model_retraining.set(
            0
        )

        return {

            "status":
                "model_promoted",

            "message":
                "New model automatically promoted to production.",

            "previous_model":
                old_info["active_model"],

            "new_model":
                new_model_filename,

            "previous_accuracy":
                old_accuracy,

            "new_accuracy":
                new_accuracy,

            "previous_f1":
                old_f1,

            "new_f1":
                new_f1,

            "drift":
                simulated_drift,

            "performance_risk":
                round(
                    performance_risk,
                    4
                ),

            "arcs":
                round(
                    arcs,
                    4
                ),

            "retraining":
                True,

            "promotion":
                True
        }

    # --------------------------------------------------------
    # REJECTION
    # --------------------------------------------------------

    else:

        model_retraining.set(
            0
        )

        return {

            "status":
                "candidate_rejected",

            "message":
                "Candidate model did not outperform production model.",

            "production_model":
                old_info["active_model"],

            "candidate_accuracy":
                new_accuracy,

            "production_accuracy":
                old_accuracy,

            "candidate_f1":
                new_f1,

            "production_f1":
                old_f1,

            "drift":
                simulated_drift,

            "performance_risk":
                round(
                    performance_risk,
                    4
                ),

            "arcs":
                round(
                    arcs,
                    4
                ),

            "retraining":
                True,

            "promotion":
                False
        }


# ============================================================
# RESET DEMO STATE
# ============================================================

@app.post("/reset")
def reset():

    global model

    # Keep V1 as the demonstration baseline.

    v1_path = os.path.join(
        MODEL_DIR,
        "model_v1.pkl"
    )

    if not os.path.exists(v1_path):

        import shutil

        shutil.copy2(
            MODEL_V1_PATH,
            v1_path
        )

    with open(
        ACTIVE_MODEL_FILE,
        "w"
    ) as f:

        json.dump(
            {
                "active_model":
                    "model_v1.pkl",
                "version":
                    "v1"
            },
            f,
            indent=4
        )

    model = load_active_model()

    model_accuracy.set(
        PRODUCTION_ACCURACY
    )

    model_precision.set(
        PRODUCTION_PRECISION
    )

    model_recall.set(
        PRODUCTION_RECALL
    )

    model_f1.set(
        PRODUCTION_F1
    )

    model_drift.set(
        0.0304
    )

    model_risk.set(
        0.0327
    )

    model_arcs.set(
        0.688
    )

    model_retraining.set(
        0
    )

    model_version.set(
        1
    )

    return {
        "status": "reset",
        "active_model": "model_v1.pkl",
        "version": "v1"
    }


# ============================================================
# PROMETHEUS
# ============================================================

@app.get("/metrics")
def metrics():

    return Response(
        generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )


# ============================================================
# WEB UI
# ============================================================

@app.get("/")
def home():

    return FileResponse(
        os.path.join(
            BASE_DIR,
            "api",
            "static",
            "index.html"
        )
    )