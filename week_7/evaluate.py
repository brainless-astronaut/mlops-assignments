import pandas as pd
import mlflow
from sklearn.metrics import accuracy_score

## ---- MLflow Setup ---- ##
mlflow.set_tracking_uri("http://127.0.0.1:5000")

## ---- Load Model and Data ---- ##
model_uri = "models:/iris-classifier/latest"

print(f"Loading model from: {model_uri}")
model = mlflow.sklearn.load_model(model_uri)

test_df = pd.read_csv("data/prepared/test.csv")
X_test, y_test = test_df.drop("target", axis = 1),  test_df["target"]

## ---- Evaluation ---- ##
predictions = model.predict(X_test)
acc = accuracy_score(y_test, predictions)

print(f"Model accuracy on test set: {acc:.4f}")
assert acc > 0.9, "Model accuracy is below 0.9 threshold!"
print("Evaluation successful!")
