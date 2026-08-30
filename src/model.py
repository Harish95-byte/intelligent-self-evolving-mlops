import joblib
import pandas as pd


MODEL_PATH = "model_v1.pkl"
PREPROCESSOR_PATH = "preprocessor_v1.pkl"


def load_model():
    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)

    return model, preprocessor


def predict(input_data):
    model, preprocessor = load_model()

    input_df = pd.DataFrame([input_data])

    processed_data = preprocessor.transform(input_df)

    prediction = model.predict(processed_data)

    return prediction[0]