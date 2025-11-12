import pandas as pd 
import mlflow, mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import yaml
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK, space_eval

## Set MLflow server URI
mlflow.set_tracking_uri("http://127.0.01:5000")
mlflow.set_experiment("Iris Experiment")

## Load Data
train_df = pd.read_csv("data/prepared/train.csv") 
test_id = pd.read_csv("data/prepared/test.csv")
X_train, y_train = train_df.drop('target', axis = 1), train_df['target']
X_test, y_test = test_df.drop('target', axis = 1), test_df['target']

## Load search space form params.yaml
params = yaml.safe_load(open('params.yaml'))['train']
max_evals = params['max_evals']

## Create Hyperopt
search_space = {}
for key, value in params['search_space'].items():
    if value['type'] == 'quniform':
        search_space[key] = hp.quniform(key, value['min'], value['max'], value['q'])
    elif value['type'] == 'uniform':
        search_space[key] = hp.uniform(key, value['min'], value['max'])

## Training function
def objective(params):
    params['n_estimators'] = int(params['n_estimators'])
    params['max_depth'] = int(params['max_depth'])

    with mlflow.start_run(nested=True):
        clf = RandomForestClassifier(**params, random_state=42)
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
        acc = accuracy_score(y_test, preds)
        
        mlflow.log_params(params)
        mlflow.log_metric('accuracy', acc)
        
        return {'loss': -acc, 'status': STATUS_OK}

## Main search
with mlflow.start_run(run_name="Hyperparameter Search") as parent_run:
    best = fmin(
        fn=objective,
        space=search_space,
        algo=tpe.suggest,
        max_evals=max_evals,
        trials=Trials()
    )
    
    # Get the best parameters
    best_params = space_eval(search_space, best)
    best_params['n_estimators'] = int(best_params['n_estimators'])
    best_params['max_depth'] = int(best_params['max_depth'])
    
    print('Best params found:', best_params)
    
    final_model = RandomForestClassifier(**best_params, random_state=42)
    final_model.fit(X_train, y_train)
    
    mlflow.log_params(best_params)
    mlflow.sklearn.log_model(
        sk_model = final_model,
        artifact_path = 'model',
        registered_model_name = 'iris-classifier' 
    )

print('Hyperopt search complete.')
