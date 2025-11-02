import os 
import mlflow
import pandas as pd
from flask import Flask, request, jsonify

## ---- MLflow Setup ---- ##
mlflow.set_tracking_uri('http://35.244.5.117:5000')

## Load the latest registered model
model_uri = 'models:/iris-classifier/latest'
model = mlflow.pyfunc.load_model(model_uri)

app = Flask(__name__)

@app.route('/predict', methods = ['POST'])
def predict():
    data = request.get_json(force = True)

    df = pd.DataFrame(data)

    prediction = model.predict(df)

    return jsonify({'prediction': list(prediction)})

if __name__ == '__main__':
    app.run(host = '0.0.0.0', port = 8080)
