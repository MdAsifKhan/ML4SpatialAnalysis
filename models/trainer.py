from sklearn.linear_model import LogisticRegression
import pickle
from utils.utils import compute_scores_train, compute_scores_test
import numpy as np
from utils.utils import graph_feature_vector, coords_to_graph, train_test_split, k_fold_split
from .factory import GraphConvolutionalNetwork
import wandb
import pandas as pd

class ModelTrainer:
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
						logfile=None):
		"""
		Initializes the ModelTrainer object.

		Args:
			config (dict): Configuration dictionary containing training parameters.
			logger (wandb.Logger): W&B logger object.
			feature_names (list, optional): List of feature names. Defaults to None.
			logfile (str, optional): Name of the log file. Defaults to None.
		"""

		self.config = config
		self.feature_names = None
		self.logfile = logfile
		self.logger = logger

		# Choose and initialize classifier based on configuration
		if self.config['name'] == 'logistic':
			self.classifier = LogisticRegression(
											random_state=self.config['seed'], 
											penalty=self.config['logisitic']['penalty'],
											solver=self.config['logisitic']['solver'],
											l1_ratio=self.config['logisitic']['l1_ratio'],
											tol=self.config['logisitic']['tol'],
											max_iter=self.config['logisitic']['max_iter']
											)
		elif self.config['name'] == 'gcn':
			self.config['gcn']['input_dim'] = self.config['feature_dim']
			self.config['gcn']['hidden_dim'] = self.config['feature_dim']
			self.classifier = GraphConvolutionalNetwork(self.config['gcn'],
														logger=self.logger)
			pass
		else:
			assert 0,f"Classifer {self.config['name']} is not implemented"

	def optimise(self, data, gtype=None):
		"""
		Optimizes the model by fitting, evaluating, and potentially saving it.

		Args:
			data (dict): A dictionary containing: expressions, enrichments (None for cell-cell case), graphs, labels, and feature names.
							expressions is a Feature matrix or a list of node attribute matrix.
			graphs (list, optional): List of graphs (for GCN models). Defaults to None.
			gtype (str, optional): Type of graph (cellcell or celltype). Defaults to None.
		"""
		data_train, data_test = train_test_split(data, 
												test_size=self.config['test_ratio'], 
												random_state=self.config['seed'])

		print(f"Fitting Model with {self.config['eval']} training")

		if self.config['eval'] == 'split':
			print(f"Fitting the {self.config['name']} Model")
			self.fit(data_train)
			y_pred_train = self.predict(data_train)

			print(f"Evaluating the {self.config['name']} Model")
			y_pred_test = self.predict(data_test)

			metrics = compute_scores_train(data_train['labels'], y_pred_train, data_test['labels'], y_pred_test)
			print(f"Saving the {self.config['name']} Model")
			self.save_model()

		elif self.config['eval'] == 'kfold':
			print(f"Fitting the {self.config['name']} Model")
			folds = k_fold_split(data, self.config['folds'], andom_state=self.config['seed'])
			y_train, y_test, y_pred_train, y_pred_test = np.array([]),np.array([]),np.array([]),np.array([])

			for i, (train_i, test_i) in enumerate(folds):
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
		else:
			assert 0, f"{self.config['eval']} Evaluation not implemented"

		self.log_metrics(metrics)
		print(metrics)

	def test(self, data):
		"""
		Tests the model on new data and returns evaluation metrics.

		Args:
			data (dict): A dictionary containing: expressions, enrichments (None for cell-cell case), graphs, labels, and feature names.
							expressions is a Feature matrix or a list of node attribute matrix.
			graphs (list, optional): List of graphs (for GCN models). Defaults to None.
			gtype (str, optional): Type of graph (cellcell or celltype). Defaults to None.
		"""
		y_pred = self.predict(X)
		return compute_scores_test(y, y_pred)

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
		if self.config['name'] == 'logistic':
			X = self.featurisation(data)
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
		if self.config['name'] == 'logistic':
			X = self.featurisation(data)
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
		if self.config['name'] == 'logistic':
			X = self.featurisation(data)
			return self.classifier.predict_proba(X)

	def featurisation(self, data, gtype='cellcell'):
		"""
		Performs feature extraction based on the specified type and criterion.

		Args:
			data (dict): A dictionary containing: expressions, enrichments (None for cell-cell case), graphs, labels, and feature names.
							expressions is a Feature matrix or a list of node attribute matrix.
			gtype (str, optional): Type of graph (cellcell or celltype). Defaults to 'cellcell'.

		Returns:
			np.ndarray: Array containing the extracted features.
		"""
		self.feature_names = data['markers']
		if gtype == 'cellcell':
			X = self.cellcell_to_featurisation(data['expressions'])
		elif gtype == 'celltype':
			X = self.celltype_featurisation(data['expressions'])
		else:
			assert 0, f"{self.config['gtype']} Expression Features are invalid"

		if self.config['gcriterion'] in ['laplacian_spectrum', 'heat_trace']:
			print(f"Computing Graph Features criterion {self.config['gcriterion']}")
			graph_features = self.graph_features(data['graphs'])
			X = np.concatenate([X, graph_features], axis=1)
			self.feature_names += [f"Graphcoeff_{i}" for i in range(graph_features.shape[1])] 

		return X

	def graph_features(self, graphs):
		"""
		Computes graph features based on the specified criterion.

		Args:
			graphs (list): List of graphs.

		Returns:
			np.ndarray: Array containing the computed graph features.
		"""
		gfeature = [graph_feature_vector(graph, self.config['gcriterion'], self.config['gf_dim']) for graph in graphs]
		return np.array(gfeature)


	def celltype_featurisation(self, X):
		"""
		Performs feature extraction based on the specified criterion for cell types.

		Args:
			X (list): List of expression data for each cell type.

		Returns:
			np.ndarray: Array containing the extracted features.
		"""
		if self.config['fcriterion'] == 'avgcelltype':
			return [expr.mean(axis=0) for expr in X]
		elif self.config['fcriterion'] == 'flatcelltype':
			data_mat = np.array(X)
			n_samples, n_celltype, n_proteins = data_mat.shape
			return data_mat.reshape(n_samples, n_celltype*n_proteins)
		else:
			assert 0,f"{self.config['fcriterion']} Not Implemented"

	def cellcell_to_featurisation(self, X):
		"""
		Performs feature extraction based on averaging expression across genes for cell-cell interactions.

		Args:
			X (list): List of expression data for each cell-cell interaction.

		Returns:
			np.ndarray: Array containing the extracted features.
		"""
		data_mat = [expr.mean(axis=0) for expr in X]
		return np.asarray(data_mat)


	def log_coefficients(self, logger):
		"""
		Logs the coefficients of the logistic regression model to W&B.

		Args:
			logger (wandb.Logger): W&B logger object.
		"""
		if self.config['name'] == 'logistic':
			coefficients = self.classifier.coef_.flatten()
			coefficients_df = pd.DataFrame({'Feature': self.feature_names, 'Coefficient': coefficients})
			logger.log({"Logistic Regression Coefficients": wandb.Table(dataframe=coefficients_df)})
			fields = {'x': 'Features', 'value': 'Coefficients'}
			logger.plot_table(data_table=table, fields=fields)
		else:
			assert 0,f"Attribution Not implemented for {self.config['name']}"

	def log_metrics(self, metrics):
		"""
		Logs the evaluation metrics to W&B.

		Args:
			metrics (dict): Dictionary containing evaluation metrics.
		"""
		metrics_table=[[key, value] for key, value in metrics.items()]
		self.logger.log({
					'Metrics': 
							wandb.Table(
									data=metrics_table, 
								columns=["Metric", "Value"])
					})

	def save_model(self, fold=None):
		"""
		Saves the trained model and feature names to a file.

		Args:
			fold (str, optional): Fold number for cross-validation (optional). Defaults to None.
		"""
		if fold:
			name = f"{self.config['name']}_{fold}"
		else:
			name = self.config['name']
		filename = f"{self.config['MODEL_PATH']}/{name}_{self.logfile}.pkl"
		out = {
				'model': self.classifier,
				'feature_names': self.feature_names
				}
		with open(filename, 'wb') as f:
			pickle.dump(out, f)

	def load_model(self, fold=None):
		"""
		Saves the trained model and feature names to a file.

		Args:
			fold (str, optional): Fold number for cross-validation (optional). Defaults to None.
		"""
		if fold:
			name = f"{self.config['name']}_{fold}"
		else:
			name = self.config['name']
		filename = f"{self.config['MODEL_PATH']}/{self.config['name']}_{self.logfile}.pkl"
		with open(filename, 'wb') as f:
			load = pickle.load(f)
			self.classifier = load['model']
			self.feature_names = load['feature_names']
