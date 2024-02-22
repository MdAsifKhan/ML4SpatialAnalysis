from sklearn.linear_model import LogisticRegression
import pickle
from sklearn.model_selection import KFold, train_test_split
from utils.utils import compute_metrics

class ModelTrainer:
	def __init__(self, config, feature_names=None):
		self.config = config
		self.feature_names = feature_names
		if self.config['name'] == 'logistic':
			self.classifier = LogisticRegression(
											random_state=self.config['seed'], 
											penalty=self.config['penalty'],
											solver=self.config['solver'],
											l1_ratio=self.config['l1_ratio'],
											tol=self.config['tol']
											)
		else:
			assert 0,f"Classifer {self.config['name']} is not implemented"

	def fit(self, X, y):
		self.classifier.fit(X, y)

	def predict(self, X):
		return self.classifier.predict(X)

	def predict_proba(self, X):
		return self.classifier.predict_proba(X)

	def save_model(self, filename):
		filename = f"{self.config['MODEL_PATH']}/{self.config['name']}_{filename}.pkl"
		out = {
				'model': self.classifier,
				'feature_names': self.feature_names
				}
		with open(filename, 'wb') as f:
			pickle.dump(out, f)

	def load_model(self, filename):
		filename = f"{self.config['MODEL_PATH']}/{self.config['name']}_{filename}.pkl"
		with open(filename, 'wb') as f:
			load = pickle.load(f)
			self.classifier = load['model']
			self.feature_names = load['feature_names']

	def train(self, X, y, filename):
		print(f"Fitting Model with {self.config['eval']} training")
		if self.config['eval'] == 'split':
			X_train, X_test, y_train, y_test = train_test_split(X, y,
												test_size=self.config['testportion'],
												random_state=self.config['seed'])

			print(f"Fitting the {self.config['name']} Model")

			self.fit(X_train, y_train)

			y_pred_train = self.predict(X_train)
			print(f"Evaluating the {self.config['name']} Model")
			y_pred_test = self.predict(X_test)
			metrics = compute_metrics(y_train, y_pred_train, y_test, y_pred_test)

			print(f"Saving the {self.config['name']} Model")
			self.save_model(filename)

		elif self.config['eval'] == 'kfold':
			print(f"Fitting the {self.config['name']} Model")
			kf = KFold(n_splits=self.config['folds'], shuffle=True, random_state=self.config['seed'])

			all_models = []
			y_train, y_test, y_pred_train, y_pred_test = [], [], [], []
			for i, (train_idx, test_idx) in enumerate(kf.split(X)):
				X_train_i, X_test_i = X[train_idx], X[test_idx]
				y_train_i, y_test_i = y[train_idx], y[test_idx]
				self.fit(X_train, y_train)

				all_y_train += y_train_i
				all_y_test += y_test_i
				all_y_pred_train += self.predict(X_train_i)
				all_y_pred_test += self.predict(X_test_i)

				self.save_model(f"{filename}_fold_{i+1}")

			print(f"Evaluating the {self.config['name']} Model")
			metrics = compute_metrics(y_train, y_pred_train, y_test, y_pred_test)

		else:
			assert 0, f"{self.config['eval']} Evaluation not implemented"

		return metrics