import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# -------------------------
# Load Dataset
# -------------------------
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42
)

# -------------------------
# Define Models
# -------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=200),
    "Random Forest": RandomForestClassifier(),
    "SVM": SVC(),
}

# -------------------------
# Create Experiment
# -------------------------
mlflow.set_experiment("iris_models")

results = []

# -------------------------
# Train & Track Models
# -------------------------
for name, model in models.items():
    with mlflow.start_run(run_name=name):
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds, average="macro")
        rec = recall_score(y_test, preds, average="macro")
        f1 = f1_score(y_test, preds, average="macro")

        # Log metrics
        mlflow.log_metrics({
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1": f1
        })

        # Log model
        mlflow.sklearn.log_model(model, name=name)

        results.append({
            "Model": name,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1": f1
        })

# -------------------------
# Collect Results
# -------------------------
results_df = pd.DataFrame(results)
print(results_df)

# -------------------------
# Select Best Model Automatically
# -------------------------
# Change this if you want to optimize for a different metric (e.g., "Accuracy")
BEST_METRIC = "F1"

best_idx = results_df[BEST_METRIC].idxmax()
best_model_name = results_df.loc[best_idx, "Model"]
best_metrics = results_df.loc[best_idx, ["Accuracy", "Precision", "Recall", "F1"]]

print(f"\nBest Model: {best_model_name}")
print("Metrics:\n", best_metrics)

# Get the trained model object
best_model = models[best_model_name]

# -------------------------
# Register the Best Model
# -------------------------
mlflow.end_run()  # ensure no run is left open

with mlflow.start_run(run_name="BestModelRegistration"):
    mlflow.log_metrics(best_metrics.to_dict())
    mlflow.sklearn.log_model(
        sk_model=best_model,
        name="IrisBestModel",
        registered_model_name="IrisBestModel",
    )
    print(f"✅ Model {best_model_name} registered successfully as IrisBestModel!")
