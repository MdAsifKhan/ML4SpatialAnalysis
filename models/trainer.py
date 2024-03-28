
import pickle
import numpy as np
from mainutils.utils import compute_scores_train, compute_scores
from mainutils.utils import leave_one_out_split, patient_level_scores
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import wandb
import pandas as pd
from torch_geometric.seed import seed_everything
import matplotlib.pyplot as plt
import io
from PIL import Image
from models.gcn import GraphConvolutionalNetwork
from models.abstract import AbstractModel

MODELS_DICT = {
	'logistic' : LogisticRegression,
	'randomforest' : RandomForestClassifier,
	'xgboost' : XGBClassifier,
	'gnn' : GraphConvolutionalNetwork,
}

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
		seed_everything(self.seed)
		# Choose and initialize classifier based on configuration
		self.config['logistic']['random_state'] = seed
		self.config['randomforest']['random_state'] = seed

		if self.config['name'] == 'gnn':
			self.config['gnn']['fnorm'] = self.config['fnorm']
			self.config['gnn']['logger'] = self.logger
			self.config[self.config['name']][self.config[self.config['name']]['gconv']]['input_dim'] = self.config['feature_dim']
			self.config[self.config['name']][self.config[self.config['name']]['gconv']] ['hidden_dim'] = 2*self.config['feature_dim']

		self.classifier = MODELS_DICT[self.config['name']](**self.config[self.config['name']])


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

		elif self.config['eval'] == 'LeaveOneOut':
			for i, (train_i, test) in enumerate(self.leave_one_out_split(data)):
				self.fit(train)
				y_pred_train_i = self.predict(train_i)
				y_pred_test_i = self.predict(test_i)

				y_train = np.concatenate([y_train, train_i['labels']])
				y_test = np.concatenate([y_test, test_i['labels']])
				y_pred_train = np.concatenate([y_pred_train, y_pred_train_i])
				y_pred_test = np.concatenate([y_pred_test, y_pred_test_i])

				self.save_model(fold=f"leaveoneout_{i+1}_patient_{test_set['patient'][0]}")

			print(f"Evaluating the {self.config['name']} Model")
			metrics = compute_scores_train(y_train, y_pred_train, y_test, y_pred_test)
			self.log_metrics(metrics, mode='LeaveOneOutROILevel')
			print('Metrics at ROI Level',metrics)
			metrics_train = patient_level_scores(y_train, y_pred_train,  data['patient'], mode='Train', pcriterion=self.config['pcriterion'])
			self.log_metrics(metrics_train, mode='LeaveOneOutPatientLevelTrain')
			print('Metrics at Patient Level', metrics_train)
			metrics_test = patient_level_scores(y_test, y_pred_test,  data['patient'], mode='Test', pcriterion=self.config['pcriterion'])
			self.log_metrics(metrics_test, mode='LeaveOneOutPatientLevelTest')
			print('Metrics at Patient Level', metrics_test)
		else:
			assert 0, f"{self.config['eval']} Evaluation not implemented"


