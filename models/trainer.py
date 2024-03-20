
import pickle
import numpy as np
from mainutils.utils import compute_scores_train, compute_scores
from mainutils.utils import graph_feature_vector, coords_to_graph, train_test_split, k_fold_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
import wandb
import pandas as pd
from torch_geometric.seed import seed_everything
import matplotlib.pyplot as plt
import io
from PIL import Image
from models.gcn import GraphConvolutionalNetwork
from models.abstract import AbstractModel

# MODELS_DICT = {
	
# }

class ModelTrainer(AbstractModel):
	"""
	This class handles training, testing, and evaluation of models.

	Attributes:
		config (dict): Configuration dictionary containing training parameters.
		logfile (str): Name of the log file (optional).
		logger (wandb.Logger): W&B logger object.
		classifier (object): Trained model classifier.
	"""
	def __init__(self, config, 
						logger,
						logfile=None,
						seed=42):
		"""
		Initializes the ModelTrainer object.

		Args:
			config (dict): Configuration dictionary containing training parameters.
			logger (wandb.Logger): W&B logger object.
			feature_names (list, optional): List of feature names. Defaults to None.
			logfile (str, optional): Name of the log file. Defaults to None.
		"""
		super().__init__(config, logger)
		self.seed = seed
		# Choose and initialize classifier based on configuration
		if self.config['name'] == 'logistic':
			self.classifier = LogisticRegression(
											random_state=seed, 
											penalty=self.config['logisitic']['penalty'],
											solver=self.config['logisitic']['solver'],
											l1_ratio=self.config['logisitic']['l1_ratio'],
											tol=self.config['logisitic']['tol'],
											max_iter=self.config['logisitic']['max_iter']
										)
		elif self.config['name'] == 'randomforest':
			self.config['randomforest']['random_state'] = seed
			self.classifier = RandomForestClassifier(**self.config['randomforest'])
		elif self.config['name'] == 'xgboost':
			self.classifier = xgb.XGBClassifier(**self.config['xgboost'])
		elif self.config['name'] == 'gcn':
			self.config['gcn']['input_dim'] = self.config['feature_dim']
			self.config['gcn']['hidden_dim'] = 2*self.config['feature_dim']
			self.config['gcn']['fnorm'] = self.config['fnorm']
			seed_everything(self.seed)
			self.classifier = GraphConvolutionalNetwork(self.config['gcn'],
														logger=self.logger)
		else:
			assert 0,f"Classifer {self.config['name']} is not implemented"

	def optimise(self, dataset):
		"""
		Optimizes the model by fitting, evaluating, and potentially saving it.

		Args:
			data (dict): A dictionary containing: expressions, enrichments (None for cell-cell case), graphs, labels, and feature names.
							expressions is a Feature matrix or a list of node attribute matrix.
			graphs (list, optional): List of graphs (for GCN models). Defaults to None.
		"""

		print(f"Fitting {self.config['name']} with {self.config['eval']} training")
		if self.config['eval'] == 'split':
			self.fit(dataset['train'])
			self.evaluate(dataset['train'], mode='Train')
			print(f"Evaluating on Test Set")
			self.evaluate(dataset['test'], mode='Test')

		elif self.config['eval'] == 'kfold':
			data = dataset['folds']
			y_train, y_test, y_pred_train, y_pred_test = np.array([]),np.array([]),np.array([]),np.array([])

			for i, (train_i, test_i) in enumerate(data):
				self.fit(train_i)
				y_pred_train_i = self.predict(train_i)
				y_pred_test_i = self.predict(test_i)

				y_train = np.concatenate([y_train, train_i['labels']])
				y_test = np.concatenate([y_test, test_i['labels']])
				y_pred_train = np.concatenate([y_pred_train, y_pred_train_i])
				y_pred_test = np.concatenate([y_pred_test, y_pred_test_i])

				self.save_model(fold=f"fold_{i+1}")

			print(f"Evaluating the {self.config['name']} Model")
			metrics = compute_scores_train(y_train, y_pred_train, y_test, y_pred_test)
			self.log_metrics(metrics, mode='kFold')
			print(metrics)
		else:
			assert 0, f"{self.config['eval']} Evaluation not implemented"

	def evaluate(self, data, mode='Test'):
		"""
		Tests the model on new data and returns evaluation metrics.

		Args:
			data (dict): A dictionary containing: expressions, enrichments (None for cell-cell case), graphs, labels, and feature names.
							expressions is a Feature matrix or a list of node attribute matrix.
			graphs (list, optional): List of graphs (for GCN models). Defaults to None.
		"""
		y_pred = self.predict(data)
		metrics = compute_scores(data['labels'], y_pred, mode)
		self.log_metrics(metrics, mode)
		print(metrics)
		if mode == 'Test':
			self.attribution(data)
