from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI(
    title="Home Credit Default Risk API"
)

model = joblib.load("models/model.pkl")


@app.get("/")
def home():
    return {
        "message": "Home Credit Default Risk Prediction API"
    }


@app.post("/predict")
def predict(data: dict):

    df = pd.DataFrame([data])

    prediction = model.predict(df)[0]

    probability = model.predict_proba(df)[0][1]

    return {
        "prediction": int(prediction),
        "default_probability": float(probability)
    }