import pandas as pd
import json
import os
import random
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

## ---- Directory Setup ---- ##
os.makedirs('data/llm', exist_ok = True)

## ---- Load Data ---- ##
print('Loading Dataset...')
data = load_iris()
df = pd.DataFrame(data.data, columns = data.feature_names)
df['target'] = data.target

df['target_name'] = df['target'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'}) # mapping targets to names 

## ---- Add 'Location' Attribute (Assignment Requirement)
random.seed(42)
df['location'] = [random.choice([0, 1]) for _ in range(len(df))]

## ---- Split Data ---- ##
train_df, val_df = train_test_split(df, test_size = 0.2, random_state = 42)

## ---- Define FOrmatting Functions ---- ##
def format_v1(row):
    '''
        V1: Raw Data representation.
        Just list the features and values.
    '''
    return (f"Sepal Length: {row['sepal length (cm)']}, "
            f"Sepal Width: {row['sepal width (cm)']}, "
            f"Petal Length: {row['petal length (cm)']}, "
            f"Petal Width: {row['petal width (cm)']}, "
            f"Location: {row['location']}")

def format_v2(row):
    '''
        V2: Descriptive representaiton.
        Uses natural language to describe the flower.
    '''
    loc_str = 'Region Alpha' if row['location'] == 0 else 'Region Beta'

    return (f"Identify the iris species. The flower has a sepal length of {row['sepal length (cm)']}cm "
            f"and a sepal width of {row['sepal width (cm)']}cm. "
            f"Its petals are {row['petal length (cm)']}cm long "
            f"and {row['petal width (cm)']}cm wide. "
            f"It was collected in {loc_str}.")

def to_gemini_json(input_text, output_text):
    '''
    Standard JSON structure for Gemini 1.5 Fine-tuning on Vertex AI.
    '''
    return {
        'contents': [
            {
                'role': 'user',
                'parts': [{'text': input_text}]
            },
            {
                'role': 'model',
                'parts': [{'text': output_text}]
            }
        ]
    }

## ---- Generate Files ---- ##
versions = {
    'v1': format_v1,
    'v2': format_v2
}

for v_name, fmt_func in versions.items():
    print(f'Generating {v_name} data...')

    with open(f'data/llm/train_{v_name}.jsonl', 'w') as f:
        for _, row in train_df.iterrows():
            json_line = to_gemini_json(fmt_func(row), row['target_name'])
            f.write(json.dumps(json_line) + '\n')

    with open(f'data/llm/val_{v_name}.jsonl', 'w') as f:
        for _, row in val_df.iterrows():
            json_line = to_gemini_json(fmt_func(row), row['target_name'])
            f.write(json.dumps(json_line) + '\n')

print('\n Success! Data generated in \'data/llm/\' directory!')
