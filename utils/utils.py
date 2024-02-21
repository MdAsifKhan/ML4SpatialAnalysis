import yaml
import numpy as np
import scipy.sparse as sp

from scipy.sparse.linalg import eigs
from scipy.sparse import csr_matrix
from sklearn.neighbors import radius_neighbors_graph, kneighbors_graph

def load_config(filename):
	with open(filename, 'r') as f:
		return yaml.safe_load(f)


def adjacency_to_laplacian(A, normalised=True):
	D = np.squeeze(np.asarray(A.sum(axis=1)))
	L = sp.diags(D) - A if sp.issparse(A) else np.diag(deg) - A
	if not normalised:
		return L
	Dsqrt = 1.0/ np.sqrt(D)
	Dsqrt[Dsqrt==np.inf] = 0
	Dsqrt = sp.diags(Dsqrt) if sp.issparse(A) else np.diag(Dsqrt)
	Lnorm = Dsqrt.dot(L).dot(Dsqrt)
	return Lnorm

def coords_to_graph(coords, method='knn', radius=7):
	if method == 'radius':
		G = radius_neighbors_graph(coords, radius, mode='connectivity',
									include_self=True)
	elif method == 'knn':
		G = kneighbors_graph(coords, radius, mode='connectivity', include_self=True)
	else:
		assert 0, f"{method} Not Implemented"
	return G

def graph_feature_vector(graph, gcriterion='degree', feature_dim=10):
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

from collections import defaultdict

def categorical_accuracy(y, y_pred):
	class_accuracy = defaultdict(int)
	class_count = defaultdict(int)

	for true_label, pred_label in zip(y, y_pred):
		class_count[true_label] += 1
		if true_label == pred_label:
			class_accuracy[true_label] += 1

	accuracy_per_class = {}
	for label in class_count.keys():
		accuracy_per_class[label] = class_accuracy[label] / class_count[label]

	return accuracy_per_class


def binary_accuracy(y, y_pred):
	return (y==y_pred).sum()