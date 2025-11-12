import os 
import mlflow
import pandas as pd
from flask import Flask, request, jsonify

## ---- MLflow Setup ---- ##
mlflow.set_tracking_uri('http://127.0.0.1:5000')

## Load the latest registered model
model_uri = 'models:/iris-classifier/latest'
model = mlflow.pyfunc.load_model(model_uri)

## Define columns
COL_ORDER = ["sepal_length", "sepal_width", "petal_length", "petal_width"]

## Create app
app = Flask(__name__)

@app.route('/predict', methods = ['POST'])
def predict():
    try:
        data = request.get_json(force = True)
        df = pd.DataFrame(data)

        df_reordered = df[COL_ORDER]

        prediction = model.predict(df)

        return jsonify({'prediction': prediction.tolist()})

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(host = '0.0.0.0', port = 8080)
