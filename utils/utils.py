import yaml
import numpy as np
import scipy.sparse as sp

from scipy.sparse.linalg import eigs
from scipy.sparse import csr_matrix
from sklearn.neighbors import radius_neighbors_graph, kneighbors_graph

def load_config(filename):
	"""
	Loads configuration from a YAML file.

	Args:
		filename (str): Path to the configuration YAML file.

	Returns:
		dict: Dictionary containing the loaded configuration.
	"""
	with open(filename, 'r') as f:
		return yaml.safe_load(f)


def adjacency_to_laplacian(A, normalised=True):
	"""
	Computes the Laplacian matrix (normalized or unnormalized) from an adjacency matrix.

	Args:
		A (numpy.ndarray or scipy.sparse.csr_matrix): Adjacency matrix of a graph.
		normalised (bool, optional): True to compute the normalized Laplacian, False for unnormalized.
		Defaults to True.

	Returns:
		Lnorm (numpy.ndarray or scipy.sparse.csr_matrix): The computed Laplacian matrix.
	"""
	D = np.squeeze(np.asarray(A.sum(axis=1)))
	L = sp.diags(D) - A if sp.issparse(A) else np.diag(deg) - A
	if not normalised:
		return L
	Dsqrt = 1.0/ np.sqrt(D)
	Dsqrt[Dsqrt==np.inf] = 0
	Dsqrt = sp.diags(Dsqrt) if sp.issparse(A) else np.diag(Dsqrt)
	Lnorm = Dsqrt.dot(L).dot(Dsqrt)
	return Lnorm

def coords_to_graph(coords, gmethod='knn', radius=7):
	"""
	Constructs a graph from coordinates using specified method (k-nearest neighbors or radius-based).

	Args:
		coords (numpy.ndarray): Array of coordinates representing nodes.
		gmethod (str, optional): Graph construction method, either 'knn' or 'radius'. Defaults to 'knn'.
		radius (float, optional): Radius for radius-based graph construction. Used only if gmethod='radius'.
									Defaults to 7.

	Returns:
		G (scipy.sparse.csr_matrix): The constructed graph adjacency matrix.
	"""
	if gmethod == 'radius':
		G = radius_neighbors_graph(coords, radius, mode='connectivity',
									include_self=True)
	elif gmethod == 'knn':
		G = kneighbors_graph(coords, radius, mode='connectivity', include_self=True)
	else:
		assert 0, f"{gmethod} Not Implemented"
	return G

def graph_feature_vector(graph, gcriterion='heat_trace', feature_dim=10):
	"""
	Extracts graph features based on provided criteria.

	Args:
		graph (scipy.sparse.csr_matrix): Adjacency matrix of the graph.
		gcriterion (str, optional): Feature extraction criterion, either 'heat_trace', or
											'laplacian_spectrum'. Defaults to 'heat_trace'.
		feature_dim (int, optional): Desired dimensionality of the feature vector. Defaults to 10.

	Returns:
		feature_vector (numpy.ndarray): Array containing the extracted graph features.
	"""
	if not isinstance(graph, csr_matrix):
		raise ValueError('nn_graph must be a scipy sparse CSR matrix.')

	num_nodes = graph.shape[0]
	feature_vector = np.zeros(feature_dim, dtype=np.float64)
	if gcriterion == 'heat_trace':
		# Compute heat trace on the graph at different timescales
		timescales = np.logspace(0, 2, num=feature_dim)

		# Compute normalized Laplacian matrix
		laplacian = adjacency_to_laplacian(graph)
		# Compute eigenvalues of normalized Laplacian
		k = min(feature_dim, num_nodes-1)
		eivals, _ = eigs(laplacian, k=k, which='SM')

		# Compute heat trace at different timescales
		for t, timescale in enumerate(timescales):
			feature_vector[t] = np.sum(np.exp(-timescale * eivals.real))
	elif gcriterion == 'laplacian_spectrum':
		# Compute normalized Laplacian matrix
		laplacian = adjacency_to_laplacian(graph)
		k = min(feature_dim, num_nodes-1)
		eivals = sp.linalg.svds(laplacian, k=k, return_singular_vectors=False)
		feature_vector = np.zeros(feature_dim)

		if len(eivals)>1:
			feature_vector[-k:] = sorted(eivals)
	else:
		assert 0, f" {gcriterion} Not implemented. Valid options are `degree`, or `heat_trace`."

	return feature_vector


from sklearn.metrics import accuracy_score, roc_auc_score, f1_score


def compute_scores_train(y_train, y_pred_train, y_test, y_pred_test):
	"""
	Computes various evaluation scores for both training and test sets.

	Args:
		y_train (list or numpy.ndarray): True labels for training set.
		y_pred_train (list or numpy.ndarray): Predicted labels for training set.
		y_test (list or numpy.ndarray): True labels for test set.
		y_pred_test (list or numpy.ndarray): Predicted labels for test set.

	Returns:
		metrics (dict): Dictionary containing evaluation metrics for both sets.
	"""
	accuracy_train = accuracy_score(y_train, y_pred_train)
	auc_train = roc_auc_score(y_train, y_pred_train)
	f1_train = f1_score(y_train, y_pred_train)

	accuracy_test = accuracy_score(y_test, y_pred_test)
	auc_test = roc_auc_score(y_test, y_pred_test)
	f1_test = f1_score(y_test, y_pred_test)
	metrics = {
			'Accuracy Train': accuracy_train,
			'Accuracy Test': accuracy_test,
			'AUC Train': auc_train,
			'AUC Test': auc_test,
			'F1 Score Train': f1_train,
			'F1 Score Test': f1_test,			
	}
	return metrics



def compute_scores_test(y, y_pred):
	"""
	Computes evaluation scores for a single test set.

	Args:
		y (list or numpy.ndarray): True labels.
		y_pred (list or numpy.ndarray): Predicted labels.

	Returns:
		metrics (dict): Dictionary containing evaluation metrics.
	"""
	accuracy_ = accuracy_score(y, y_pred)
	auc_ = roc_auc_score(y, y_pred)
	f1_ = f1_score(y, y_pred)

	metrics = {
			'Accuracy': accuracy_,
			'AUC': auc_,
			'F1 Score': f1_,
	}
	return metrics
