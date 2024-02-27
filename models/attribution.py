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

	def log_shap_scores(self, X):
		"""
		Computes SHAP scores for a given data set.

		Args:
		X (np.ndarray): Feature matrix.

		Returns:
		np.ndarray: Array of SHAP scores.
		"""

		if self.clf_name == 'LogisticRegression':
			explainer = shap.KernelExplainer(self.classifier.predict_proba, X)
			shap_values = explainer.shap_values(X)
			logger.log({'Shapley Score':shap_values})
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
