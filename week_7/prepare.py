import os
import sys
import yaml
import pandas as pd
from sklearn.model_selection import train_test_split

# Read params
params = yaml.safe_load(open("params.yaml"))["prepare"]
split = params["split"]
seed = params["seed"]

# Read input file path from command line
input_file = sys.argv[1]

# Create output directory
os.makedirs("data/prepared", exist_ok=True)

# Read the data, rename columns, and map text labels to numbers
df = pd.read_csv(input_file)
df.rename(columns={
    "sepal.length": "sepal_length",
    "sepal.width": "sepal_width",
    "petal.length": "petal_length",
    "petal.width": "petal_width",
    "variety": "target"
}, inplace=True)

# Map string targets to integers
target_map = {"Setosa": 0, "Versicolor": 1, "Virginica": 2}
df['target'] = df['target'].map(target_map)

# Split and save
train, test = train_test_split(df, test_size=split, random_state=seed)
train.to_csv("data/prepared/train.csv", index=False)
test.to_csv("data/prepared/test.csv", index=False)
