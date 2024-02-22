from sklearn.linear_model import LogisticRegression



class Model:
	def __init__(self, config, seed=42):
		self.config = config
		if self.config['name'] == 'logistic':
			self.classifier = LogisticRegression(random_state=seed, 
											penalty=self.config['penalty'],
											solver=self.config['solver'],
											l1_ratio=self.config['l1_ratio'],
											tol=self.config['tol'])
		else:
			assert 0,f"Classifer {self.config['name']} is not implemented"

	def forward(self):
		return None
