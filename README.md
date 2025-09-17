MLOPS-Assignment-1

Model Registration with MLflow

After training multiple models (Logistic Regression, Random Forest, and SVM) on the Iris dataset, we log all experiments and metrics in MLflow Tracking Server.

Steps Performed

Run Experiments
Each model is trained, evaluated, and logged with MLflow including metrics (Accuracy, Precision, Recall, F1) and the serialized model.

Select Best Model
We automatically select the best-performing model based on the F1-score. (This metric can be changed to Accuracy/Precision/Recall if needed.)

Register Model in MLflow Registry
The best model is registered in MLflow Model Registry under the name IrisBestModel.
MLflow automatically assigns a version number each time a new best model is registered (v1, v2, etc.).

🔧 How to View Registered Models

Run the MLflow UI:

mlflow ui


Then open http://127.0.0.1:5000 in your browser.

Go to Experiments → iris_models to see all runs and metrics.

Navigate to Models → IrisBestModel to see all registered versions.

You can track lineage, metrics, and artifacts of each registered version.