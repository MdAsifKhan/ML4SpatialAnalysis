import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression
import wandb
from models.abstract import AbstractModel


class ModelAttribution(AbstractModel):
	"""
	This class performs model attribution using SHAP for logistic regression models.

	Attributes:
		config (dict): Configuration dictionary containing training parameters.
		classifier (object): Trained logistic regression model.
		feature_names (list): List of feature names.
		clf_name (str): Name of the classifier type.
	"""
	def __init__(self, config, logger=None):
		"""
		Initializes the ModelAttribution object.

		Args:
			config (dict): Configuration dictionary containing attribution parameters.
		"""
		super().__init__(config, logger)
		filename = f"{self.config['MODEL_PATH']}/{self.config['name']}_{self.logfile}.pkl"
		with open(filename, 'wb') as f:
			load = pickle.load(f)
			self.classifier = load['model']
			self.feature_names = load['feature_names']
			self.scaler = load['scaler']

	def run_attribution(self, data):
		"""
		Logs the coefficients of the logistic regression model to W&B.

		Args:
			logger (wandb.Logger): W&B logger object.
		"""

		self.attribution(data)
