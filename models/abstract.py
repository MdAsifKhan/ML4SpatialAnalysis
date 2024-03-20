import pickle
import numpy as np
from mainutils.utils import graph_feature_vector
import wandb
import matplotlib.pyplot as plt
import io
from PIL import Image
from abc import ABC
from sklearn.preprocessing import StandardScaler


class AbstractModel(ABC):
	"""
		Abstract class for handling common functionalities of model training.

		Attributes:
		config (dict): Configuration dictionary containing training parameters.
		logger (wandb.Logger): W&B logger object (optional).
	"""

	def __init__(self, config, logger=None):
		self.config = config
		self.logger = logger
		self.feature_names = None
		self.scaler = None

	def cellcell_to_featurisation(self, X):
		"""
		Performs feature extraction based on averaging expression across genes for cell-cell contact.

		Args:
			X (list): List of expression data for each cell-cell interaction.

		Returns:
			np.ndarray: Array containing the extracted features.
		"""
		if self.config['fnorm'] == 'raw':
			data_mat = np.asarray([expr.mean(axis=0) for expr in X])
		elif self.config['fnorm'] == 'znorm':
			data_mat = np.asarray([((expr - np.mean(expr, axis=0, keepdims=True))/(1e-8 + np.std(expr, axis=0, keepdims=True))).mean(axis=0) for expr in X])
		elif self.config['fnorm'] == 'log1p':
			data_mat = np.asarray([np.log1p(expr).mean(axis=0) for expr in X])
		elif self.config['fnorm'] == 'minmax':
			data_mat = np.asarray([((expr - expr.min(axis=0))/(expr.max(axis=0) - expr.min(axis=0) + 1e-8)).mean(axis=0) for expr in X])
		elif self.config['fnorm'] is None:
			data_mat = np.array([])
		else:
			assert 0, f"{self.config['fnorm']} Expression Features are invalid"

		return data_mat


	def graph_features(self, graphs):
		"""
		Computes graph features based on the specified criterion.

		Args:
			graphs (list): List of graphs.

		Returns:
			np.ndarray: Array containing the computed graph features.
		"""
		gfeature_all = []
		for graph in graphs:
			gfeature, gfname = graph_feature_vector(graph, self.config['gcriterion'], self.config['gf_dim']) 
			gfeature_all.append(gfeature)
		return np.array(gfeature_all), gfname

	def fit(self, data):
		"""
		Fits the model to the training data.

		Args:
			X (np.ndarray): Feature matrix or a list of node attribute matrix.
			y (np.ndarray): Labels.
			graphs (list, optional): List of graphs (for GCN models). Defaults to None.
		"""
		if self.config['name'] == 'gcn':
			self.classifier.fit(data)
			return
		if self.config['name'] in ['logistic', 'randomforest', 'xgboost']:
			X = self.cellcell_to_featurisation(data['expressions'])
			if self.config['normalise_features']:
				self.scaler = StandardScaler()
				self.scaler.fit(X)
				X = self.scaler.transform(X)
			self.classifier.fit(X, data['labels'])
			return 

	def predict(self, data):
		"""
		Predicts labels for new data points.

		Args:
			X (np.ndarray): Feature matrix or a list of node attribute matrix.
			graphs (list, optional): List of graphs (for GCN models). Defaults to None.

		Returns:
			np.ndarray: Predicted labels.
		"""
		if self.config['name'] == 'gcn':
			return self.classifier.predict(data)
		if self.config['name'] in ['logistic', 'randomforest', 'xgboost']:
			X = self.cellcell_to_featurisation(data['expressions'])
			if self.config['normalise_features']:
				X = self.scaler.transform(X)
			return self.classifier.predict(X)

	def predict_proba(self, data):
		"""
		Predicts class probabilities for new data points.

		Args:
			X (np.ndarray): Feature matrix or a list of node attribute matrix.
			graphs (list, optional): List of graphs (for GCN models). Defaults to None.

		Returns:
			np.ndarray: Predicted class probabilities.
		"""
		if self.config['name'] == 'gcn':
			return self.classifier.predict_proba(data)
		if self.config['name'] in ['logistic', 'randomforest', 'xgboost']:
			X = self.cellcell_to_featurisation(data['expressions'])
			if self.config['normalise_features']:
				X = self.scaler.transform(X)
			return self.classifier.predict_proba(X)

	def attribution(self, data):
		"""
		Computes attribution scores for a given data set.

		Args:
			data (np.ndarray): A dictionary containing: expressions, enrichments (None for cell-cell case), graphs, labels, and feature names.
							expressions is a Feature matrix or a list of node attribute matrix.
		"""

		if self.config['name'] == 'gcn':
			self.classifier.pyg_attribution(data)
			self.classifier.gradient_attribution(data)
		elif self.config['name'] in ['logistic', 'randomforest', 'xgboost']:
			if self.config['name'] == 'logistic':
				feature_importances = self.classifier.coef_.flatten()
			else:
				feature_importances = self.classifier.feature_importances_
			feature_names = data['markers']
			sorted_indices = feature_importances.argsort()[::-1][:10]
			sorted_feature_importances = feature_importances[sorted_indices]
			sorted_feature_names = np.array(feature_names)[sorted_indices]
			plt.figure(figsize=(16, 10))
			plt.bar(range(len(sorted_feature_importances)), sorted_feature_importances, tick_label=sorted_feature_names)
			plt.xlabel('Proteins', fontsize=28)
			plt.ylabel('Importance Score', fontsize=28)
			plt.title(f"Logistic Regression Classifer", fontsize=32)
			plt.xticks(rotation=45, ha='right', fontsize=28)
			plt.subplots_adjust(bottom=0.2)
			plt.tight_layout()
			buffer = io.BytesIO()
			buffer.seek(0)
			plt.savefig(buffer, format='png')
			self.logger.log({f"Importance scores {self.config['name']} classifer": wandb.Image(Image.open(buffer))})
		else:
			assert 0,f"Attribution not implemented for {self.config['name']}"


	def save_model(self, logname, fold=None):
		"""
		Saves the trained model and feature names to a file.

		Args:
			fold (str, optional): Fold number for cross-validation (optional). Defaults to None.
		"""
		if fold:
			name = f"{self.config['name']}_{fold}"
		else:
			name = self.config['name']
		filename = f"{self.config['LOG_PATH']}/{name}_{logname}.pkl"
		out = {
				'model': self.classifier,
				'scaler': self.scaler
				}
		with open(filename, 'wb') as f:
			pickle.dump(out, f)


	def log_metrics(self, metrics, mode='Train'):
		"""
		Logs the evaluation metrics to W&B.

		Args:
			metrics (dict): Dictionary containing evaluation metrics.
		"""
		metrics_table=[[key, value] for key, value in metrics.items()]
		self.logger.log({
					f"{mode} Metrics": 
							wandb.Table(
									data=metrics_table, 
								columns=['Metric', 'Value'])
					})
