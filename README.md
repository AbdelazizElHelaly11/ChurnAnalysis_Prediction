# Telco Customer Churn Prediction and Analysis

This project aims to predict customer churn for a telecom company using machine learning techniques and end-to-end deployment. It was developed as part of the **AI Data Science Track – Digital Egypt Pioneers Initiative**.

We use the [Telco Customer Churn dataset from Kaggle](https://www.kaggle.com/datasets/yeanzc/telco-customer-churn-ibm-dataset) and implement the full data science lifecycle — from data cleaning and model building to deployment and monitoring.

---

## 📁 Project Files

- `DEPI_GRAD_PROJECT_with_pipeline.ipynb`: Jupyter notebook with all steps of the ML pipeline
- `Churn_Analysis_Report.pdf`: Detailed project report including visuals, model comparisons, deployment architecture, and test cases
- `app.py`: Streamlit dashboard for interactive predictions
- `churn_api.py`: FastAPI prediction endpoint
- `monitoring.py`: MLflow-based monitoring script

---

## 🎯 Objectives

- Analyze factors driving customer churn
- Develop predictive models using classification algorithms
- Compare model performance based on accuracy, precision, recall, F1, and AUC
- Deploy the best model with FastAPI
- Build an interactive dashboard with Streamlit
- Monitor predictions and trigger retraining with MLflow

---

## 📊 Dataset

- **Name:** Telco Customer Churn Dataset  
- **Source:** [Kaggle](https://www.kaggle.com/datasets/yeanzc/telco-customer-churn-ibm-dataset)

The dataset includes information about customer demographics, account details, service usage, and whether they churned.

---

## 🧠 Models Trained

| Model                | Test Accuracy | F1 Score | Notes                     |
|---------------------|---------------|----------|---------------------------|
| Random Forest        | 0.8691        | 0.8711   | Highest performing model  |
| XGBoost              | 0.8599        | 0.8624   | Balanced performance      |
| Gradient Boosting    | 0.8401        | 0.8444   | Solid and reliable        |
| Logistic Regression  | 0.8106        | 0.8200   | Good baseline             |
| Naive Bayes          | 0.8092        | 0.8209   | Interpretable model       |
| Neural Network       | 0.7623        | 0.8046   | Best recall (96%)         |

---

## ⚙️ Deployment Pipeline

### Core Components:
- **API:** FastAPI (`churn_api.py`)
- **Dashboard:** Streamlit (`app.py`)
- **Monitoring:** MLflow (`monitoring.py`)
- **Model Serving:** Serialized with `joblib`/`pickle`

### `/predict` Endpoint
Accepts customer details in JSON and returns:
- Churn prediction (`0` or `1`)
- Probability score


## 📚 References

- Scikit-learn
- XGBoost
- FastAPI
- Streamlit
- MLflow
- Matplotlib
- Seaborn
- NumPy
- Pandas

---

## 🧑‍💻 Authors

Abdelaziz Elhelaly, Ahmad Mohammed, Abdelrahman Khaled,  
Mazen Hesham, Norhan Momen, Mahmoud Ahmad  
**AI Data Science Track – Digital Egypt Pioneers Initiative**
