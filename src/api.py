from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

app = FastAPI(
    title="Home Credit Default Risk Prediction API",
    description="MLOps Project"
)

# Load model
model = joblib.load('models/model.pkl')
print("Model loaded successfully - 39 features")

class ClientData(BaseModel):
    NAME_CONTRACT_TYPE: float
    CODE_GENDER: float
    FLAG_OWN_CAR: float
    FLAG_OWN_REALTY: float
    AMT_INCOME_TOTAL: float
    AMT_CREDIT: float
    NAME_INCOME_TYPE: float
    NAME_EDUCATION_TYPE: float
    NAME_FAMILY_STATUS: float
    NAME_HOUSING_TYPE: float
    DAYS_BIRTH: float
    DAYS_EMPLOYED: float
    DAYS_REGISTRATION: float
    OWN_CAR_AGE: float
    OCCUPATION_TYPE: float
    CNT_FAM_MEMBERS: float
    ORGANIZATION_TYPE: float
    EXT_SOURCE_1: float
    EXT_SOURCE_2: float
    EXT_SOURCE_3: float
    LIVINGAREA_MODE: float
    FONDKAPREMONT_MODE: float
    HOUSETYPE_MODE: float
    TOTALAREA_MODE: float
    BUREAU_DAYS_CREDIT_MIN: float
    BUREAU_DAYS_CREDIT_MAX: float
    BUREAU_DAYS_CREDIT_MEAN: float
    BUREAU_CREDIT_DAY_OVERDUE_MEAN: float
    BUREAU_AMT_CREDIT_SUM_SUM: float
    BUREAU_AMT_CREDIT_SUM_MEAN: float
    BUREAU_AMT_CREDIT_SUM_OVERDUE_MEAN: float
    GOODS_INCOME_RATIO: float
    EXT_SOURCE_RANGE: float
    ANNUITY_INCOME_RATIO: float
    PAYMENT_RATE: float
    GOODS_CREDIT_RATIO: float
    EXT_MEAN: float
    EXT_MIN: float
    EXT_MAX: float


@app.get("/")
def home():
    return {"message": "Home Credit Default Risk Prediction API is running!"}

@app.post("/predict")
def predict(data: ClientData):
    try:
        input_dict = data.model_dump()
        input_df = pd.DataFrame([input_dict])

        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]

        risk_level = "HIGH RISK (Default Likely)" if prediction == 1 else "LOW RISK"

        return {
            "prediction": int(prediction),
            "risk_level": risk_level,
            "default_probability": round(float(probability), 4)
        }
    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)