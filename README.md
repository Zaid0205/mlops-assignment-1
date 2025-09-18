MLOPS-Assignment-1

# Iris MLflow Experiment — README

## Overview

This repository contains an example MLOps workflow using the Iris dataset. The project trains three models (Logistic Regression, Random Forest, SVM), tracks experiments with **MLflow**, automatically selects the best model (by F1-score), and registers it in the MLflow Model Registry.

---

## Problem statement & dataset description

**Problem statement**: Create a supervised classification pipeline for the Iris dataset, evaluate multiple models, choose the best performing model based on a target metric, and register that model with MLflow for reproducibility and deployment.

**Dataset**: Iris dataset (4 features: sepal length, sepal width, petal length, petal width; 3 target classes). The dataset is the built-in `sklearn.datasets.load_iris()`.

---

## Project structure

```
MLOPS-ASSIGNMENT-1/
├─ .gitignore
├─ README.md                
├─ requirements.txt         
├─ results/
│  └─ mlflow_model_results.csv
├─ models/                    
├─ mlruns/                    (MLflow logs)
└─ src/
   └─ mlflow_tracking.py
   └─ train.py     <-- training + logging + registration
```


## Model selection & comparison

**Models trained**:

* Logistic Regression (solver `lbfgs`, `C=0.5`, `max_iter=200`)
* Random Forest (`n_estimators=50`, `max_depth=5`)
* SVM (`kernel='rbf'`, `C=2`, `gamma='scale'`)

**Metrics logged**: Accuracy, Precision, Recall, F1

**How best model is chosen**: The script computes F1 for each model and picks the one with the highest F1. The selected model is then registered in MLflow Model Registry.

---

## MLflow logging (what we log)

* Parameters/hyperparameters for each model
* Metrics: Accuracy, Precision, Recall, F1
* Artifacts: confusion matrix images, results CSV (`../results/mlflow_model_results.csv`)
* Serialized model for each run (logged via `mlflow.sklearn.log_model`)

**Experiment name**: `iris_models` (set in `mlflow_tracking.py`)

## Model registration

* The script registers the best model under the name `IrisBestModel` in the MLflow Model Registry.
* Each time you re-run the pipeline and the best model is registered, MLflow increments the version number.

**How to load the registered model for inference**:

```python
import mlflow.pyfunc
# load by model name and version (example version 1):
model = mlflow.pyfunc.load_model("models:/IrisBestModel/1")
# or load latest:
model_latest = mlflow.pyfunc.load_model("models:/IrisBestModel/Production")  # if promoted

preds = model.predict(new_data)
```

---

## Instructions to run the code (step-by-step)

1. Clone the repo:

   ```bash
   git clone <your-repo-url>
   cd mlops-assignment-1
   ```

2. Create a virtual environment and install dependencies:

   ```bash
   python -m venv .venv
   # Windows PowerShell
   .\.venv\Scripts\Activate.ps1
   # or Git Bash / Linux
   source .venv/bin/activate

   pip install --upgrade pip
   pip install -r requirements.txt
   ```

   **Minimum `requirements.txt`** example (add to your repo):

   ```text
   numpy
   pandas
   scikit-learn
   matplotlib
   seaborn
   mlflow
   joblib
   ```

3. Start MLflow UI (optional but recommended):

   ```bash
   mlflow ui --port 5000
   ```

   Open [http://127.0.0.1:5000](http://127.0.0.1:5000) in your browser to inspect runs and models.

4. Run training & logging script (from project root):

   ```bash
   cd src
   python mlflow_tracking.py
   ```

   This will:

   * Train 3 models
   * Log parameters, metrics, and artifacts to MLflow (local `mlruns/` by default)
   * Save `../results/mlflow_model_results.csv`
   * Register the best model in MLflow Model Registry as `IrisBestModel`

5. Check the MLflow UI to view runs and artifacts. Check **Models → IrisBestModel** to see the registered versions.

---

## How to capture and include MLflow screenshots

1. Start the MLflow UI (see above).

2. Open these pages in your browser:

   * Experiment runs page for `iris_models` (for run comparison)
   * Model Registry page for `IrisBestModel` (to show the registered version)

3. Take screenshots and save them with filenames:

   * `mlflow_runs.png`
   * `model_registry.png`

## Git & pushing (recommended guidelines)

**Recommendation**: *Do not* push heavy MLflow logs (`mlruns/`) or large serialized models to the repository. Instead:

* Keep your code and `README.md` in Git
* Keep `results/mlflow_model_results.csv` and `docs/images/*` if needed for reproducibility/visuals
* Add `mlruns/` to `.gitignore` to prevent accidental commits

**If you want to push only code, docs and results**:

```bash
# from repo root
git add src/mlflow_tracking.py README.md requirements.txt results/mlflow_model_results.csv docs/images/*
git commit -m "Add MLflow tracking script, README, results and screenshots"
git push origin main
```

**If you *must* push the full `mlruns/` folder (NOT recommended)**:

```bash
# ensure mlruns/ is not in .gitignore or force-add
git add -f mlruns/
git commit -m "Add MLflow run logs (mlruns)"
git push origin main
```

**If mlruns/ was previously committed and you want to remove it from history**:

```bash
git rm -r --cached mlruns
echo "mlruns/" >> .gitignore
git add .gitignore
git commit -m "Remove mlruns from tracking and ignore it"
git push origin main
```

## Appendix: Quick commands cheat-sheet

```bash
cd src
python mlflow_tracking.py

mlflow ui --port 5000

git add src/mlflow_tracking.py README.md results/mlflow_model_results.csv docs/images/*
git commit -m "Add training script, README & results"
git push origin main
```

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