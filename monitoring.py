import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd
import matplotlib.pyplot as plt

# Connect to MLflow server
client = MlflowClient()

# Get all runs from the default experiment
experiment = client.get_experiment_by_name("Default")
runs = client.search_runs(experiment_ids=[experiment.experiment_id])

# Extract churn probabilities
data = []
for run in runs:
    if 'probability_of_churn' in run.data.metrics:
        data.append({
            'Run Name': run.info.run_name,
            'Churn Probability': run.data.metrics['probability_of_churn']
        })

if data:
    df = pd.DataFrame(data)
    
    # Save monitoring results
    df.to_csv('monitoring_metrics.csv', index=False)

    # Plot
    plt.figure(figsize=(10,5))
    plt.plot(df['Churn Probability'], marker='o')
    plt.title('Churn Probabilities Over Predictions')
    plt.xlabel('Prediction Number')
    plt.ylabel('Churn Probability')
    plt.grid()
    plt.show()

    # Monitoring simple alert
    avg_churn = df['Churn Probability'].mean()
    if avg_churn > 0.7:
        print("⚠️ Warning: Average churn probability is high! Consider retraining.")
    else:
        print(f"✅ Average churn probability is healthy: {avg_churn:.2f}")

else:
    print("No runs found with 'probability_of_churn' metric.")
