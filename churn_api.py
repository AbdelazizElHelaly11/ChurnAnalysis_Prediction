from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import mlflow
from fastapi.responses import JSONResponse

app = FastAPI()

# Load the trained model once
model = joblib.load("churn_pipeline.pkl")

# Define input schema
class InputData(BaseModel):
    Partner: int
    Dependents: int
    Tenure_Months: int
    Online_Security: int
    Online_Backup: int
    Device_Protection: int
    Tech_Support: int
    Streaming_TV: int
    Streaming_Movies: int
    Contract: int
    Monthly_Charges: float
    Total_Charges: float
    CLTV: int
    Internet_Service_Fiber_optic: int
    Internet_Service_No: int

# Helper function to rename columns
def rename_columns(df):
    return df.rename(columns={
        "Tenure_Months": "Tenure Months",
        "Online_Security": "Online Security",
        "Online_Backup": "Online Backup",
        "Device_Protection": "Device Protection",
        "Tech_Support": "Tech Support",
        "Streaming_TV": "Streaming TV",
        "Streaming_Movies": "Streaming Movies",
        "Monthly_Charges": "Monthly Charges",
        "Total_Charges": "Total Charges",
        "Internet_Service_Fiber_optic": "Internet Service_Fiber optic",
        "Internet_Service_No": "Internet Service_No",
    })

# Helper function to log MLflow data
def log_to_mlflow(params, churn_probability):
    with mlflow.start_run(run_name="Churn_Prediction"):
        mlflow.log_params(params)
        mlflow.log_metric("probability_of_churn", churn_probability)

@app.post("/predict/")
def predict(input_data: InputData):
    try:
        # Convert input to DataFrame
        df = pd.DataFrame([input_data.dict()])
        df = rename_columns(df)

        # Make prediction
        prediction = model.predict(df)[0]
        churn_probability = model.predict_proba(df)[0][1]

        # Log to MLflow
        log_to_mlflow(input_data.dict(), churn_probability)

        # Return prediction
        return {
            "prediction": int(prediction),
            "probability_of_churn": churn_probability
        }

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
