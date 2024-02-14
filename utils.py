import yaml
import numpy as np
import scipy.sparse as sp

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

from scipy.sparse.linalg import eigs
from scipy.sparse import csr_matrix

def graph_feature_vector(graph, feature_type='degree', num_time_scales=10):
    if not isinstance(graph, csr_matrix):
        raise ValueError('nn_graph must be a scipy sparse CSR matrix.')
    
    num_nodes = knn_graph.shape[0]
    
    if feature_type == 'degree':
        # Compute node degrees
        feature_vector = knn_graph.sum(axis=1).A.ravel()
    elif feature_type == 'heat_trace':
        # Compute heat trace on the graph at different timescales
        timescales = np.logspace(0, 2, num=num_time_scales)
        feature_vector = np.zeros(num_time_scales, dtype=np.float64)
        
        # Compute normalized Laplacian matrix
        laplacian = adjacency_to_laplacian(graph)
        # Compute eigenvalues of normalized Laplacian
        eivals, _ = eigs(laplacian, k=min(num_nodes-1, num_time_scales), which='SM')
        
        # Compute heat trace at different timescales
        for t, timescale in enumerate(timescales):
            feature_vector[t] = np.sum(np.exp(-timescale * eivals.real))
    
    else:
        raise ValueError('Invalid feature_type. Valid options are `degree`, or `heat_trace`.')
    
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