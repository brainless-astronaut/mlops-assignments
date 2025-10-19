import os
import sys
import yaml
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from joblib import dump

# Read params
params = yaml.safe_load(open("params.yaml"))["train"]
seed = params["seed"]
n_est = params["n_est"]
min_split = params["min_split"]

# Read input/output paths from command line
input_dir = sys.argv[1]
output_dir = sys.argv[2]

# Create output directory for models
os.makedirs(output_dir, exist_ok=True)

# Load data
train_df = pd.read_csv(os.path.join(input_dir, "train.csv"))

X_train = train_df.drop("target", axis=1)
y_train = train_df["target"]

# Train model
clf = RandomForestClassifier(
    n_estimators=n_est, min_samples_split=min_split, random_state=seed
)
clf.fit(X_train, y_train)

# Save model
dump(clf, os.path.join(output_dir, "model.pkl"))
