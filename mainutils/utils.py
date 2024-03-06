import yaml
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigs
from scipy.sparse import csr_matrix
from sklearn.neighbors import radius_neighbors_graph, kneighbors_graph
import numpy as np
import networkx as nx

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

		feature_names = []
		# Compute heat trace at different timescales
		for t, timescale in enumerate(timescales):
			feature_vector[t] = np.sum(np.exp(-timescale * eivals.real))
			feature_names.append(f"_HeatTrace_{timescale:.3f}")

	elif gcriterion == 'laplacian_spectrum':
		# Compute normalized Laplacian matrix
		laplacian = adjacency_to_laplacian(graph)
		k = min(feature_dim, num_nodes-1)
		eivals = sp.linalg.svds(laplacian, k=k, return_singular_vectors=False)
		feature_vector = np.zeros(feature_dim)
		if len(eivals)>1:
			feature_vector[-k:] = sorted(eivals)
		feature_names = [f"_eigen_{i}" for i in range(feature_dim)]

	elif gcriterion == 'graphproperties':
		num_edges = graph.getnnz() / 2.0  # Divide by 2 since the matrix is symmetric
		num_nodes = graph.shape[0]
		density = num_edges / (num_nodes * (num_nodes - 1) / 2)  # Complete graph denominator

		average_degree = np.mean(np.sum(graph != 0, axis=0))
		graphnx = nx.from_numpy_array(graph.toarray())
		clustering_coefficient = nx.average_clustering(graphnx)
		avg_path = nx.average_shortest_path_length(graphnx)
		#connectivity = 1.0 if nx.is_connected(graphnx) else 0.0

		feature_vector = np.array([num_nodes, num_edges, density, clustering_coefficient, average_degree])
		feature_names = ['_num_nodes', '_num_edges', '_density', '_clustcoeff', '_avgdegree']
	else:
		assert 0, f" {gcriterion} Not implemented. Valid options are `degree`, or `heat_trace`."

	return feature_vector, feature_names


from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, f1_score
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
	balanced_accuracy_train = balanced_accuracy_score(y_train, y_pred_train)
	auc_train = roc_auc_score(y_train, y_pred_train)
	f1_train = f1_score(y_train, y_pred_train)

	accuracy_test = accuracy_score(y_test, y_pred_test)
	balanced_accuracy_test = balanced_accuracy_score(y_test, y_pred_test)
	auc_test = roc_auc_score(y_test, y_pred_test)
	f1_test = f1_score(y_test, y_pred_test)
	metrics = {
			'Accuracy Train': accuracy_train,
			'Balanced_Accuracy Train': balanced_accuracy_train,
			'Accuracy Test': accuracy_test,
			'Balanced_Accuracy Test': balanced_accuracy_test,
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
	balanced_accuracy_ = balanced_accuracy_score(y, y_pred)
	auc_ = roc_auc_score(y, y_pred)
	f1_ = f1_score(y, y_pred)

	metrics = {
			'Accuracy': accuracy_,
			'Balanced_Accuracy': balanced_accuracy_,
			'AUC': auc_,
			'F1 Score': f1_,
	}
	return metrics



def train_test_split(dataset, test_size=0.2, random_state=None):
	"""
	Custom train-test split for a dictionary-like dataset.

	Args:
		dataset (dict): Dictionary where keys represent different data matrices or labels.
		test_size (float): Ratio of the dataset to include in the test set.
		random_state (int or None): Random seed for reproducibility.

	Returns:
		train_set (dict): Dictionary containing train split for each key.
		test_set (dict): Dictionary containing test split for each key.
	"""
	np.random.seed(random_state)

	nm_samples =  len(dataset['labels'])

	patient_ids = dataset['patient']
	unique_patient_ids = np.unique(patient_ids)
	test_size = int(test_size * len(unique_patient_ids))

	indices = np.random.permutation(unique_patient_ids)

	train_patient_ids = unique_patient_ids[test_size:]
	test_patient_ids = unique_patient_ids[:test_size]

	train_indices = [i for i, patient in enumerate(patient_ids) if patient in train_patient_ids]
	test_indices = [i for i, patient in enumerate(patient_ids) if patient in test_patient_ids]

	train_set = {}
	test_set = {}

	for key, data in dataset.items():
		if data is None:
			train_set[key] = None
			test_set[key] = None
		elif key == 'markers':
			train_set[key] = data
			test_set[key] = data
		else:
			train_set[key] = [data[i] for i in train_indices]
			test_set[key] = [data[i] for i in test_indices]
	return train_set, test_set

import numpy as np

def k_fold_split(dataset, k=5, random_state=None):
	"""
	Custom k-fold split for a dictionary-like dataset.

	Args:
		dataset (dict): Dictionary where keys represent different data matrices or labels.
		k (int): Number of folds.
		random_state (int or None): Random seed for reproducibility.

	Returns:
		fold_sets (list): List of k fold sets, where each fold set is a tuple containing train and test splits for each key.
	"""
	np.random.seed(random_state)
	num_samples = len(dataset['labels'])
	unique_ids, inverse_indices = np.unique(np.array(dataset['patient']), return_inverse=True)

	fold_indices = np.array_split(np.random.permutation(len(unique_ids)), k)

	fold_sets = []
	for fold_idx in range(k):
		test_unique_ids = unique_ids[fold_indices[fold_idx]]
		train_unique_ids = unique_ids[np.concatenate([fold_indices[i] for i in range(k) if i != fold_idx])]

		test_indices = np.where(np.isin(dataset['patient'], test_unique_ids))[0]
		train_indices = np.where(np.isin(dataset['patient'], train_unique_ids))[0]

		train_set = {}
		test_set = {}

		for key, data in dataset.items():
			if data is None:
				train_set[key] = None
				test_set[key] = None
			elif key == 'markers':
				train_set[key] = data
				test_set[key] = data
			else:
				train_set[key] = [data[i] for i in train_indices]
				test_set[key] = [data[i] for i in test_indices]

		fold_sets.append((train_set, test_set))
	return fold_sets