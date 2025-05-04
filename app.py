import streamlit as st
import pandas as pd
import pickle
import altair as alt

st.title("🔮 Customer Churn Prediction")

# ✅ تحميل الموديل (باستخدام st.cache_resource بدلاً من st.cache)
@st.cache_resource
def load_model():
    with open("random_forest_model.pkl", "rb") as file:
        model = pickle.load(file)
    return model

model = load_model()

# 🖊️ إدخال بيانات المستخدم
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
    "Internet_Service_Fiber_optic": get_input("Internet_Service_Fiber_optic (إنترنت ألياف ضوئية)", ["No", "Yes"]),
    "Internet_Service_No": get_input("Internet_Service_No (بدون خدمة إنترنت)", ["Yes", "No"])
}

# 🚀 التوقع
if st.button("🔍 Predict"):
    try:
        input_df = pd.DataFrame([inputs])
        prediction = model.predict(input_df)[0]
        proba = model.predict_proba(input_df)[0][1]
        stay = 1 - proba

        st.subheader(f"Stay Probability: {stay:.2%}")
        
        if prediction == 0:
            st.success("✅ Will Stay")
        else:
            st.error("⚠️ Will Churn")

        chart = alt.Chart(pd.DataFrame({
            "Status": ["Churn", "Stay"],
            "Probability": [proba, stay]
        })).mark_bar().encode(
            x="Status", y="Probability", color="Status"
        )

        st.altair_chart(chart, use_container_width=True)

    except Exception as e:
        st.error(f"Error: {e}")
