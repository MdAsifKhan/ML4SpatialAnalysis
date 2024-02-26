from sklearn.linear_model import LogisticRegression
import pickle
from sklearn.model_selection import KFold, train_test_split
from utils.utils import compute_scores_train, compute_scores_test
import numpy as np
from utils.utils import graph_feature_vector, coords_to_graph
from .factory import tnbcGCN
import wandb

class ModelTrainer:
	"""
	This class handles training, testing, and evaluation of models.

	Attributes:
		config (dict): Configuration dictionary containing training parameters.
		feature_names (list): List of feature names (optional).
		logfile (str): Name of the log file (optional).
		logger (wandb.Logger): W&B logger object.
		classifier (object): Trained model classifier.
	"""
	def __init__(self, config, 
						logger,
						feature_names=None,
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
		self.feature_names = feature_names
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
			self.config['gcn']['input_dim'] = len(feature_names)
			self.config['gcn']['hidden_dim'] = len(feature_names)
			self.classifier = tnbcGCN(self.config['gcn'],
									logger=self.logger)
			pass
		else:
			assert 0,f"Classifer {self.config['name']} is not implemented"

	def fit(self, X, y, graphs=None):
		"""
		Fits the model to the training data.

		Args:
			X (np.ndarray): Feature matrix or a list of node attribute matrix.
			y (np.ndarray): Labels.
			graphs (list, optional): List of graphs (for GCN models). Defaults to None.
		"""
		if self.config['name'] == 'gcn':
			self.classifier.fit(X, y, graphs)
			return
		self.classifier.fit(X, y)

	def predict(self, X, graphs=None):
		"""
		Predicts labels for new data points.

		Args:
			X (np.ndarray): Feature matrix or a list of node attribute matrix.
			graphs (list, optional): List of graphs (for GCN models). Defaults to None.

		Returns:
			np.ndarray: Predicted labels.
		"""
		if self.config['name'] == 'gcn':
			return self.classifier.predict(X, graphs)
		return self.classifier.predict(X)

	def predict_proba(self, X, graphs=None):
		"""
		Predicts class probabilities for new data points.

		Args:
			X (np.ndarray): Feature matrix or a list of node attribute matrix.
			graphs (list, optional): List of graphs (for GCN models). Defaults to None.

		Returns:
			np.ndarray: Predicted class probabilities.
		"""
		if self.config['name'] == 'gcn':
			return self.classifier.predict_proba(X, graphs)
		return self.classifier.predict_proba(X)

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

	def featurisation(self, X, gtype='cellcell'):
		"""
		Performs feature extraction based on the specified type and criterion.

		Args:
			X (list): List of expression data.
			gtype (str, optional): Type of graph (cellcell or celltype). Defaults to 'cellcell'.

		Returns:
			np.ndarray: Array containing the extracted features.
		"""
		if gtype == 'cellcell':
			e_features = self.cellcell_to_featurisation(X)
		elif gtype == 'celltype':
			e_features = self.celltype_featurisation(X)
		else:
			assert 0, f"{self.config['gtype']} Expression Features are invalid"
		return e_features

	def optimise(self, X, y, graphs=None, gtype=None):
		"""
		Optimizes the model by fitting, evaluating, and potentially saving it.

		Args:
			X (np.ndarray): Feature matrix or a list of node attribute matrix.
			y (np.ndarray): Labels.
			graphs (list, optional): List of graphs (for GCN models). Defaults to None.
			gtype (str, optional): Type of graph (cellcell or celltype). Defaults to None.
		"""
		if self.config['gcriterion'] != 'gcn':
			X = self.featurisation(X, gtype)

		if self.config['gcriterion'] in ['laplacian_spectrum', 'heat_trace']:
			print(f"Computing Graph Features criterion {self.config['gcriterion']}")
			graph_features = self.graph_features(graphs)
			X = np.concatenate([X, graph_features], axis=1)
			self.feature_names += [f"Graphcoeff_{i}" for i in range(graph_features.shape[1])] 

		print(f"Fitting Model with {self.config['eval']} training")

		if self.config['eval'] == 'split':
			if self.config['gcriterion'] == 'gcn':
				X_train, X_test, y_train, y_test, graphs_train, graphs_test = train_test_split(X, y, graphs,
																				test_size=self.config['testportion'],
																				random_state=self.config['seed'])

			else:
				X_train, X_test, y_train, y_test = train_test_split(X, y,
													test_size=self.config['testportion'],
													random_state=self.config['seed'])
				graphs_train, graphs_test = None, None

			print(f"Fitting the {self.config['name']} Model")
			self.fit(X_train, y_train, graphs_train)
			y_pred_train = self.predict(X_train, graphs_train)

			print(f"Evaluating the {self.config['name']} Model")
			y_pred_test = self.predict(X_test, graphs_test)

			metrics = compute_scores_train(y_train, y_pred_train, y_test, y_pred_test)
			print(f"Saving the {self.config['name']} Model")
			self.save_model()

		elif self.config['eval'] == 'kfold':
			print(f"Fitting the {self.config['name']} Model")
			kf = KFold(n_splits=self.config['folds'], shuffle=True, random_state=self.config['seed'])

			y_train, y_test, y_pred_train, y_pred_test = np.array([]),np.array([]),np.array([]),np.array([])
			for i, (train_idx, test_idx) in enumerate(kf.split(X)):
				X_train_i, X_test_i = X[train_idx], X[test_idx]
				y_train_i, y_test_i = y[train_idx], y[test_idx]
				if self.config['gcriterion'] == 'gcn':
					if graphs_train and graphs_test:
						graphs_train_i = graphs_train[train_idx]
						graphs_test_i = graphs_test[test_idx]
					else:
						assert 0,"Graphs is None"
				else:
					graphs_train_i, graphs_test_i = None, None
				self.fit(X_train_i, y_train_i, graphs_train_i)

				y_train = np.concatenate([y_train, y_train_i])
				y_test = np.concatenate([y_test, y_test_i])

				y_pred_train_i = self.predict(X_train_i, graphs_train_i)
				y_pred_test_i = self.predict(X_test_i, graphs_test_i)

				y_pred_train = np.concatenate([y_pred_train, self.predict(y_pred_train_i)])
				y_pred_test = np.concatenate([y_pred_test, self.predict(y_pred_test_i)])

				self.save_model(fold=f"fold_{i+1}")

			print(f"Evaluating the {self.config['name']} Model")
			metrics = compute_scores_train(y_train, y_pred_train, y_test, y_pred_test)
		else:
			assert 0, f"{self.config['eval']} Evaluation not implemented"

		self.log_metrics(metrics)

	def test(self, X, y, graphs=None):
		"""
		Tests the model on new data and returns evaluation metrics.

		Args:
			X (np.ndarray): Feature matrix or a list of node attribute matrix.
			y (np.ndarray): Labels.
			graphs (list, optional): List of graphs (for GCN models). Defaults to None.
			gtype (str, optional): Type of graph (cellcell or celltype). Defaults to None.
		"""
		if self.config['use_graph_features']:
			print(f"Computing Graph Features criterion {self.config['gcriterion']}")
			graph_features = self.graph_features(graphs)
			X = np.concatenate([X, graph_features], axis=1)
		y_pred = self.predict(X)
		return compute_scores_test(y, y_pred)

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

