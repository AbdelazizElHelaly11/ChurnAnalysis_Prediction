import streamlit as st
import pandas as pd
import altair as alt
import pickle
import os
import mlflow
from mlflow.tracking import MlflowClient

# ✅ Improved model loading with absolute path
@st.cache_resource
def load_model():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "random_forest_model.pkl")
    
    try:
        with open(model_path, "rb") as file:
            model = pickle.load(file)
            return model
    except FileNotFoundError:
        st.error(f"❌ Model file not found at: {model_path}")
        raise

model = load_model()

st.title("Customer Churn Prediction")

# User Inputs
st.sidebar.header("📥 Customer Info")

def get_input(label, options=["Yes", "No"]):
    return 1 if st.sidebar.selectbox(label, options) == "Yes" else 0

inputs = {
    "Partner": get_input("Partner (شريك)"),
    "Dependents": get_input("Dependents (أطفال أو تابعين)"),
    "Tenure_Months": st.sidebar.number_input("Tenure_Months (مدة الاشتراك)", 0, 72),
    "Online_Security": get_input("Online_Security (أمان الإنترنت)"),
    "Online_Backup": get_input("Online_Backup (نسخ احتياطي أونلاين)"),
    "Device_Protection": get_input("Device_Protection (حماية الأجهزة)"),
    "Tech_Support": get_input("Tech_Support (دعم فني)"),
    "Streaming_TV": get_input("Streaming_TV (مشاهدة تلفزيون)"),
    "Streaming_Movies": get_input("Streaming_Movies (مشاهدة أفلام)"),
    "Contract": st.sidebar.selectbox(
        "Contract (نوع العقد: 0=شهري, 1=سنة, 2=سنتين)", [0, 1, 2]
    ),
    "Monthly_Charges": st.sidebar.number_input("Monthly_Charges (الرسوم الشهرية)", 18, 120),
    "Total_Charges": st.sidebar.number_input("Total_Charges (الإجمالي المدفوع)", 20, 9000),
    "CLTV": st.sidebar.number_input("CLTV (قيمة العميل)", 2000, 7000),
    "Internet Service_Fiber optic": get_input("Internet Service_Fiber optic (إنترنت ألياف ضوئية)", ["No", "Yes"]),
    "Internet Service_No": get_input("Internet Service_No (بدون خدمة إنترنت)", ["Yes", "No"])
}

# Helper function to rename columns to match model expectations
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
        "Internet Service_Fiber optic": "Internet Service_Fiber optic",
        "Internet Service_No": "Internet Service_No"
    })

if st.button("🔍 Predict"):
    try:
        # Convert and rename columns
        input_df = pd.DataFrame([inputs])
        input_df = rename_columns(input_df)
        
        # Make prediction
        pred = model.predict(input_df)[0]
        churn_prob = model.predict_proba(input_df)[0][1]
        stay_prob = 1 - churn_prob

        st.subheader(f"Stay Probability: {stay_prob:.2%}")

        if pred == 0:
            st.success("✅ Will Stay")
        else:
            st.error("⚠ Will Churn")

        # Visualization
        chart = alt.Chart(pd.DataFrame({
            "Status": ["Churn", "Stay"],
            "Probability": [churn_prob, stay_prob]
        })).mark_bar().encode(
            x="Status", y="Probability", color="Status"
        )

        st.altair_chart(chart, use_container_width=True)

    except Exception as e:
        st.error(f"🚨 Prediction Error: {str(e)}")

# MLflow monitoring section
st.header("Model Monitoring")

if st.button("🔄 Refresh Monitoring Data"):
    try:
        client = MlflowClient()
        experiment = client.get_experiment_by_name("Default")
        runs = client.search_runs(experiment_ids=[experiment.experiment_id])

        data = []
        for run in runs:
            if 'probability_of_churn' in run.data.metrics:
                data.append({
                    'Run Name': run.info.run_name,
                    'Churn Probability': run.data.metrics['probability_of_churn']
                })

        if data:
            df = pd.DataFrame(data)
            df.to_csv('monitoring_metrics.csv', index=False)
            
            st.line_chart(df.set_index('Run Name')['Churn Probability'])
            
            avg_churn = df['Churn Probability'].mean()
            if avg_churn > 0.7:
                st.warning("⚠️ Warning: High average churn probability! Consider retraining.")
            else:
                st.success(f"✅ Healthy average churn probability: {avg_churn:.2f}")
        else:
            st.info("ℹ️ No monitoring data found")

    except Exception as e:
        st.error(f"Monitoring Error: {str(e)}")
