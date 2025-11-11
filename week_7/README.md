## MLOps Assignment 5

### What's in This Project? 
Here is a list of the important files and what each one does:

 - prepare.py: This is our data cleaner. It takes the raw iris.csv data, gives the columns proper names, and splits it into a training set and a testing set.

 - train.py: This is our model trainer. It uses the clean data to train many different versions of our model (hyperparameter tuning) and records all the results in our MLflow "lab notebook."

 - evaluate.py: This is our final exam script. It downloads the best model from our MLflow library (the Model Registry) and runs one last test to make sure it works well.

 - dvc.yaml: This is the main recipe book for our project. It tells DVC the exact steps to run the pipeline, like running prepare.py first and then train.py.

 - params.yaml: This file holds all the important settings for our project, like which model settings to try out during training.

 - requirements.txt: This is the shopping list for our project. It tells Python which libraries (like pandas and mlflow) need to be installed for the code to work.
