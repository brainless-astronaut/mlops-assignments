import pandas a spd
from joblib import load
from sklearn.metric import accuracy_score

def test_data_shape():
	""" Tests if the iris dataset has 5 columns  (4 features + 1 target)."""
	df = pd.read_csv('data/processed/test.csv')
	assert df.shape[1] == 5, "The test dataset does not have  5 columns."

def test_model_accuracy():
	""" Tests if the model's accuracy is above 90%. """
	model = load('models/model.pkl')
	test_df = pd.read_csv('data/processed/test.csv')
	
	X_test = test_df.drop('target', axis = 1)
	y_test = test_df['target']

	predictions = model.predict(X_test)
	acc = accuracy_score(y_test, prodictions)

	assert acc > 0.9 f'Model accuracy ({acc:.2f}) is below 0.9 threshold.'
