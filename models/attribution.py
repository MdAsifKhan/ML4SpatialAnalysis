import pickle
import numpy as np
import shap
from sklearn.linear_model import LogisticRegression
import wandb


class ModelAttribution:
	"""
	This class performs model attribution using SHAP for logistic regression models.

	Attributes:
		config (dict): Configuration dictionary containing training parameters.
		classifier (object): Trained logistic regression model.
		feature_names (list): List of feature names.
		clf_name (str): Name of the classifier type.
	"""
	def __init__(self, config):
		"""
		Initializes the ModelAttribution object.

		Args:
			config (dict): Configuration dictionary containing attribution parameters.
		"""
		self.config = config
		with open(f"{self.config['MODEL_PATH']}/{self.config['filename']}", 'wb') as f:
			load = pickle.load(f)
			self.classifier = load['model']
			self.feature_names = load['feature_names']

		self.clf_name = type(self.classifier).__name__

	def shap_scores(self, X, y):
		"""
		Computes SHAP scores for a given data set.

		Args:
		X (np.ndarray): Feature matrix.
		y (np.ndarray): Labels.

		Returns:
		np.ndarray: Array of SHAP scores.
		"""

		if self.clf_name == 'LogisticRegression':
			explainer = shap.KernelExplainer(self.classifier.predict_proba, X)
			shap_values = explainer.shap_values(X)
			return shap_values
		else:
			assert 0,f"Attribution Not implemented for {self.clf_name}"

	def log_coefficients(self, logger):
		"""
		Logs the coefficients of the logistic regression model to W&B.

		Args:
			logger (wandb.Logger): W&B logger object.
		"""
		if self.clf_name == 'LogisticRegression':
			coefficients = self.classifier.coef_.flatten()
			coefficients_df = pd.DataFrame({'Feature': self.feature_names, 'Coefficient': coefficients})
			logger.log({"Logistic Regression Coefficients": wandb.Table(dataframe=coefficients_df)})
			fields = {'x': 'Features', 'value': 'Coefficients'}
			logger.plot_table(data_table=table, fields=fields)
		else:
			assert 0,f"Attribution Not implemented for {self.clf_name}"
