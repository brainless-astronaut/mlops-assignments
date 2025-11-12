import os
import sys
import yaml
import pandas as pd
from sklearn.model_selection import train_test_split

## Read params
params = yaml.safe_load(open("params.yaml"))["prepare"]
split = params["split"]
seed = params["seed"]

## Create output directory
os.makedirs("data/prepared", exist_ok = True)

## Define columns
col_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'target']

## Reading the data
df = pd.read_csv('data/raw/iris.csv', header = None, names = col_names)

## Mapping the string tagets to integers
target_map = {'Iris-setosa': 0, 'Iris-versicolor': 1, "Iris-virginica': 2}
df['target'] = df['target'].map(target_map)

## Split and save
train, test = train_test_split(df, test_size=split, random_state=seed)
train.to_csv("data/prepared/train.csv", index=False)
test.to_csv("data/prepared/test.csv", index=False)
