from sklearn.linear_model import LogisticRegression



class ModelTrainer:
	def __init__(self, config):
		self.config = config
		if self.config['name'] == 'logistic':
			self.classifier = LogisticRegression(random_state=self.config['seed'], 
													penalty=self.config['penalty'],
													solver=self.config['solver'])
		else:
			assert 0,f"Classifer {self.config['name']} is not implemented"

	def fit(self, X, y):
		self.classifier.fit(X, y)

	def predict(self, X):
		return self.classifier.predict(X)

	def predict_proba(self, X):
		return self.classifier.predict_proba(X)

	def accuracy(self, X, y):
		return self.classifier.score(X, y)

	def confidence_scores(self, X):
		return self.classifier.decision_function(X)