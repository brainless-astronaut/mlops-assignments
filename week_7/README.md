# MLOps Week 7: Deployment and Autoscaling

This project builds a complete CI/CD pipeline to deploy and autoscale an Iris species classifier.

## File Utility

* **`prepare.py`**: Cleans raw data and splits it into `train.csv` and `test.csv`.
* **`train.py`**: Uses Hyperopt to find the best model and logs all experiments and the final model to MLflow.
* **`evaluate.py`**: A script to load the "latest" registered model from MLflow and confirm it works.
* **`app.py`**: A Flask web server that loads the registered model and serves predictions.
* **`requirements.txt`**: A list of all Python libraries needed for this project.
* **`params.yaml`**: Contains all parameters for the DVC pipeline and the Hyperopt search space.
* **`dvc.yaml`**: The DVC "recipe" that defines the `prepare` and `train` stages.
* **`Dockerfile`**: Instructions for Docker to package the `app.py` into a container.
* **`deployment.yaml`**: Instructions for Kubernetes to run our app and expose it to the internet.
* **`hpa.yaml`**: Instructions for Kubernetes to automatically scale our app from 1 to 3 pods based on CPU load.
* **`script.lua`**: A helper script for the `wrk` stress-testing tool, defining the JSON data for `POST` requests.
* **`.github/workflows/main.yml`**: The master GitHub Actions workflow that automatically builds, deploys, and stress-tests the application on every push to `main`.
