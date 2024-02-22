import pickle
import numpy as np
import shap
from sklearn.linear_model import LogisticRegression
import wandb


class ModelAttribution:
	def __init__(self, config):
		self.config = config
		with open(f"{self.config['MODEL_PATH']}/{self.config['filename']}", 'wb') as f:
			load = pickle.load(f)
			self.classifier = load['model']
			self.feature_names = load['feature_names']

		self.clf_name = type(self.classifier).__name__

	def shap_scores(self, X, y):
		return None

	def log_coefficients(self, logger):
		if self.clf_name == 'LogisticRegression':
			coefficients = self.classifier.coef_.flatten()
			coefficients_df = pd.DataFrame({'Feature': self.feature_names, 'Coefficient': coefficients})
			logger.log({"Logistic Regression Coefficients": wandb.Table(dataframe=coefficients_df)})
			fields = {'x': 'Features', 'value': 'Coefficients'}
			logger.plot_table(data_table=table, fields=fields)
		else:
			assert 0,f"Attribution Not implemented for {self.clf_name}"

