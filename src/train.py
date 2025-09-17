import pandas as pd
import joblib
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split dataset
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)

# Logistic Regression
log_reg = LogisticRegression(max_iter=200)
log_reg.fit(x_train, y_train)
pred1 = log_reg.predict(x_test)

acc1 = accuracy_score(y_test, pred1)
prec1 = precision_score(y_test, pred1, average="macro")
rec1 = recall_score(y_test, pred1, average="macro")
f11 = f1_score(y_test, pred1, average="macro")

joblib.dump(log_reg, "../models/log_model.pkl")

# Random Forest
forest = RandomForestClassifier(n_estimators=20, random_state=0)
forest.fit(x_train, y_train)
pred2 = forest.predict(x_test)

acc2 = accuracy_score(y_test, pred2)
prec2 = precision_score(y_test, pred2, average="macro")
rec2 = recall_score(y_test, pred2, average="macro")
f12 = f1_score(y_test, pred2, average="macro")

joblib.dump(forest, "../models/forest_model.pkl")

# SVM
svm_model = SVC(kernel="rbf")
svm_model.fit(x_train, y_train)
pred3 = svm_model.predict(x_test)

acc3 = accuracy_score(y_test, pred3)
prec3 = precision_score(y_test, pred3, average="macro")
rec3 = recall_score(y_test, pred3, average="macro")
f13 = f1_score(y_test, pred3, average="macro")

joblib.dump(svm_model, "../models/svm_model.pkl")

# Compile results
results = [
    ["Logistic Regression", acc1, prec1, rec1, f11],
    ["Random Forest", acc2, prec2, rec2, f12],
    ["SVM", acc3, prec3, rec3, f13]
]

df = pd.DataFrame(results, columns=["Model", "Accuracy", "Precision", "Recall", "F1"])
print(df)

# Save results
df.to_csv("../results/model_results.csv", index=False)
