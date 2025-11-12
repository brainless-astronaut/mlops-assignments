import os
import sys
import yaml
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK, space_eval

# --- 1. Set MLflow Server URI ---
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Iris Model Tuning")

# --- 2. Load Data ---
params_all = yaml.safe_load(open("params.yaml"))
params = params_all["train"]
seed = params["seed"]
max_evals = params["max_evals"]

input_dir = sys.argv[1]
output_dir = sys.argv[2]

os.makedirs(output_dir, exist_ok=True)

train_df = pd.read_csv(os.path.join(input_dir, "train.csv"))
test_df = pd.read_csv(os.path.join(input_dir, "test.csv"))
X_train = train_df.drop("target", axis=1)
y_train = train_df["target"]
X_test = test_df.drop("target", axis=1)
y_test = test_df["target"]

# --- 3. Create Hyperopt search space from YAML ---
search_space = {}
for key, value in params["search_space"].items():
    if value["type"] == "quniform":
        search_space[key] = hp.quniform(key, value["min"], value["max"], value["q"])
    elif value["type"] == "uniform":
        search_space[key] = hp.uniform(key, value["min"], value["max"])

# 4. Define the training function (objective)
def objective(params):
    # Cast params to int
    params['n_estimators'] = int(params['n_estimators'])
    params['max_depth'] = int(params['max_depth'])

    with mlflow.start_run(nested=True):
        clf = RandomForestClassifier(**params, random_state=seed)
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
        acc = accuracy_score(y_test, preds)

        mlflow.log_params(params)
        mlflow.log_metric("accuracy", acc)

        return {'loss': -acc, 'status': STATUS_OK}

# 5. Run the main search
with mlflow.start_run(run_name="Hyperparameter Search") as parent_run:
    best = fmin(
        fn=objective,
        space=search_space,
        algo=tpe.suggest,
        max_evals=max_evals,
        trials=Trials()
    )

    best_params = space_eval(search_space, best)
    best_params['n_estimators'] = int(best_params['n_estimators'])
    best_params['max_depth'] = int(best_params['max_depth'])

    print("Best params found:", best_params)

    # Train the final model on all data
    final_model = RandomForestClassifier(**best_params, random_state=seed)
    final_model.fit(X_train, y_train)

    # Log and register the final model
    mlflow.log_params(best_params)
    mlflow.sklearn.log_model(
        sk_model=final_model,
        artifact_path="model",
        registered_model_name="iris-classifier"
    )
    print("Hyperopt search complete and model registered.")
