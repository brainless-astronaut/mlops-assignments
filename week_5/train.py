import os
import sys
import yaml
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn

## --- MLflow Setup --- ##
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Iris Model Tuning")

## --- Load Data and Params --- ##
params = yaml.safe_load(open("params.yaml"))["train"]
seed = params["seed"]

input_dir = sys.argv[1]
output_dir = sys.argv[2]

os.makedirs(output_dir, exist_ok=True)

train_df = pd.read_csv(os.path.join(input_dir, "train.csv"))
test_df = pd.read_csv(os.path.join(input_dir, "test.csv"))

X_train = train_df.drop("target", axis=1)
y_train = train_df["target"]
X_test = test_df.drop("target", axis=1)
y_test = test_df["target"]

## --- Hyperparameter Tuning Loop --- ##
for n_est in params["n_est"]:
    for min_split in params["min_split"]:
        with mlflow.start_run():
            mlflow.log_param("n_estimators", n_est)
            mlflow.log_param("min_samples_split", min_split)

            clf = RandomForestClassifier(
                n_estimators=n_est, min_samples_split=min_split, random_state=seed
            )
            clf.fit(X_train, y_train)

            predictions = clf.predict(X_test)
            acc = accuracy_score(y_test, predictions)

            mlflow.log_metric("accuracy", acc)

            mlflow.sklearn.log_model(clf, "model")

            print(f"Logged run with n_est={n_est}, min_split={min_split}, accuracy={acc:.4f}")
