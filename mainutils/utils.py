import yaml
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigs
from scipy.sparse import csr_matrix
from sklearn.neighbors import radius_neighbors_graph, kneighbors_graph, NearestNeighbors
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from scipy.sparse import csr_matrix
from matplotlib.cm import ScalarMappable
import matplotlib.patches as patches
from collections import Counter
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, f1_score
from scipy.spatial.distance import pdist, cdist
from scipy.spatial import Delaunay
from scipy.spatial import KDTree

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


def edge_index_to_adj(edge_index, num_nodes):
	row = edge_index[0].cpu().numpy()
	col = edge_index[1].cpu().numpy()
	data = np.ones_like(row)
	adj = csr_matrix((data, (row, col)), shape=(num_nodes, num_nodes))
	return adj


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


def delaunay_graph(coords, mode='connectivity'):
	"""
		Constructs a Delaunay graph based on given coordinates.

	Args:
		coords: A NumPy array of shape (n_points, n_dims) containing coordinates.
		mode: Either 'connectivity' (for adjacency matrix) or 'distance' (for distance matrix).
				Defaults to 'connectivity'.

	Returns:
		A scipy.sparse.csr_matrix (connectivity mode) or a NumPy array (distance mode)
		representing the Delaunay graph.
	"""
	nm_nodes = len(coords)
	triangluation = Delaunay(coords)
	indptr, indices = triangluation.vertex_neighbor_vertices
	if mode=='connectivity':
		return csr_matrix((np.ones_like(indices, dtype=np.float64), indices, indptr), shape=(nm_nodes, nm_nodes))
	if mode=='distance':
		distances = cdist(coords, coords)
		A = distances[indptr[:-1], indices].reshape(nm_nodes, -1)
		return csr_matrix(A)


def atmostk_neighbors_graph(coords, k, mode='connectivity', include_self=False):
	"""
		Constructs an "at-most-k" nearest neighbor graph based on Euclidean distances.

	Args:
		coords: A NumPy array of shape (n_samples, 2) representing data points.
		k: The maximum number of neighbors allowed for each vertex (at most k).

	Returns:
		A scipy.sparse.csr_matrix representing the adjacency matrix of the "at-most-k" nearest neighbor graph.
	"""
	# Construct a k-nearest neighbor graph
	tree = KDTree(coords)
	indices = tree.query(coords, k + 1)[1][:, 1:]  # Exclude the first nearest neighbor (itself)

	G = nx.Graph()
	for i, neighbors in enumerate(indices):
		# Add self-loop if specified
		if include_self:
			G.add_edge(i, i, weight=1)

		for j in neighbors:
			dist = np.linalg.norm(coords[i] - coords[j])
			if mode == 'distance':
				G.add_edge(i, j, weight=1.0/(dist + 1e-6))
			else:
				G.add_edge(i, j, weight=1)

	# Sparsify based on threshold
	threshold = np.percentile([G[i][j]['weight'] for i, j in G.edges()], 100 * (1 - k / len(coords)), interpolation='higher')
	edges_to_remove = [(i, j) for i, j in G.edges() if G[i][j]['weight'] > threshold]
	G.remove_edges_from(edges_to_remove)

	# Ensure each node is connected to at least one neighbor
	for i in range(len(coords)):
		if len(list(G.neighbors(i))) == 0:
			j = indices[i][0]  # Connect to the nearest neighbor
			dist = np.linalg.norm(coords[i] - coords[j])
			if mode == 'distance':
				G.add_edge(i, j, weight=1.0/(dist + 1e-6))
			else:
				G.add_edge(i, j, weight=1)
	return nx.adjacency_matrix(G)


def coords_to_graph(coords, gmethod='knn', radius=7):
	"""
	Constructs a graph from coordinates using specified method (k-nearest neighbors, k-atmost neighbors or radius-based).

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
	elif gmethod == 'atmostk':
		G = atmostk_neighbors_graph(coords, radius, mode='connectivity', include_self=True)
	elif gmethod == 'delaunay':
		G = delaunay_graph(coords, mode='connectivity')
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
		timescales = np.logspace(-2, 2, num=feature_dim)

		# Compute normalized Laplacian matrix
		laplacian = adjacency_to_laplacian(graph)
		# Compute eigenvalues of normalized Laplacian
		k = min(feature_dim, num_nodes-2)
		eivals, _ = eigs(laplacian, k=k, which='SM')

		feature_names = []
		# Compute heat trace at different timescales
		for t, timescale in enumerate(timescales):
			feature_vector[t] = np.sum(np.exp(-timescale * eivals.real))
			feature_names.append(f"_HeatTrace_{t:.3f}")

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
		#avg_path = nx.average_shortest_path_length(graphnx)
		connectivity = 1.0 if nx.is_connected(graphnx) else 0.0

		feature_vector = np.array([num_nodes, num_edges, density, clustering_coefficient, average_degree, connectivity])
		feature_names = ['_num_nodes', '_num_edges', '_density', '_clustcoeff', '_avgdegree', '_connectivity']
	else:
		assert 0, f" {gcriterion} Not implemented. Valid options are `degree`, or `heat_trace`."

	return feature_vector, feature_names


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



def compute_scores(y, y_pred, y_proba, mode='Train', multi_class='raise'):
	"""
	Computes evaluation scores for a single test set.

	Args:
		y (list or numpy.ndarray): True labels.
		y_pred (list or numpy.ndarray): Predicted labels.

	Returns:
		metrics (dict): Dictionary containing evaluation metrics.
	"""
	accuracy_ = accuracy_score(y, y_pred)
	auc_ = roc_auc_score(y, y_proba, multi_class=multi_class)
	f1_ = f1_score(y, y_pred, average='weighted')
	bal_acc_ = balanced_accuracy_score(y, y_pred)
	metrics = {
			f"{mode} Accuracy": accuracy_,
			f"{mode} AUC": auc_,
			f"{mode} F1 Score": f1_,
			f"{mode} Balanced Accuracy": bal_acc_,

	}
	return metrics



def patient_level_scores(y, y_pred, y_proba, patients, mode='Test', pcriterion='majority'):
	unique_patients = list(set(patients))
	patients_preds = {patient : [] for patient in unique_patients}
	patients_labels = {patient : [] for patient in unique_patients}
	patients_probs = {patient : [] for patient in unique_patients}
	true_positives = 0
	true_negatives = 0
	false_positives = 0
	false_negatives = 0

	for patient, label, pred, prob in zip(patients, y, y_pred, y_proba):
		patients_preds[patient].append(pred)
		patients_probs[patient].append(prob)
		patients_labels[patient].append(label)

	unique_pred_patients_prob, unique_pred_patients_label, unique_patients_label = [], [], []
	for patient in unique_patients:
		correct_predictions = [1 if pred == label == 1 else 0 for pred, label in zip(patients_preds[patient], patients_labels[patient])]
		if pcriterion == 'majority':
			# Check if the majority of predictions match the majority of labels
			vote_patient = Counter(correct_predictions).most_common(1)[0][0]
			prob_patient = np.mean(patients_probs[patient])
		else:
			assert 0,f"{pcriterion} Not Implemented"

		# Assign patient as true positive or true negative based on majority correct predictions
		if vote_patient == 1:
			unique_pred_patients_label.append(1)
		else:
			unique_pred_patients_label.append(0)
		unique_patients_label.append(patients_labels[patient][0])
		unique_pred_patients_prob.append(prob_patient)

	aucroc = roc_auc_score(unique_patients_label, unique_pred_patients_prob)
	accuracy = accuracy_score(unique_patients_label, unique_pred_patients_label)
	f1_ = f1_score(unique_patients_label, unique_pred_patients_label)
	bal_acc_ = balanced_accuracy_score(unique_patients_label, unique_pred_patients_label)

	metrics = {
			f"{mode} Accuracy {pcriterion}": accuracy,
			f"{mode} AUC {pcriterion}": aucroc,
			f"{mode} F1 Score {pcriterion}": f1_,
			f"{mode} Balanced Accuracy {pcriterion}": bal_acc_,
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
	unique_patient_ids = np.random.permutation(unique_patient_ids)

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
		elif key == 'markers' or key=='celltypes':
			train_set[key] = data
			test_set[key] = data
		else:
			train_set[key] = [data[i] for i in train_indices]
			test_set[key] = [data[i] for i in test_indices]

	return train_set, test_set


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


def leave_one_out_split(data):
	unique_patients = list(set(data['patient']))

	for leave_out_patient in unique_patients:
		print(f"Leave One Out Validation on a patient {leave_out_patient}")
		train_patients = [patient for patient in unique_patients if patient != leave_out_patient]
		train_idx = [idx for idx, patient in enumerate(data['patients']) if patient != leave_out_patient]
		test_idx = [idx for idx, patient in enumerate(data['patients']) if patient == leave_out_patient]

		train_set, test_set = {}, {}
		for key, data in data.items():
			if data is None:
				train_set[key] = None
				test_set[key] = None
			elif key in ['markers', 'celltypes']:
				train_set[key] = data
				test_set[key] = data
			else:
				train_set[key] = [data[i] for i in train_idx]
				test_set[key] = [data[i] for i in test_set]

		yield train_set, test_set


def feature_normalisation(X, fnorm):
	if fnorm == 'raw':
		return X
	if fnorm == 'znorm':
		return [((expr - np.mean(expr, axis=0, keepdims=True))/(1e-8 + np.std(expr, axis=0, keepdims=True))) for expr in X]
	if fnorm == 'log1p':
		X_norm = []
		for expr in X:
			expr_s = np.log1p(expr)
			expr_ns = (expr_s - np.mean(expr_s, axis=0, keepdims=True))/(1e-8 + np.std(expr_s, axis=0, keepdims=True))
			X_norm.append(expr_ns)
		return X_norm
	if fnorm == 'minmax':
		X_norm = []
		for expr in X:
			expr_s = np.log1p(expr)
			expr_ns = (expr - expr.min(axis=0))/(expr.max(axis=0) - expr.min(axis=0) + 1e-8)
			X_norm.append(expr_ns)
		return X_norm
	if fnorm == 'arctan':
		X_norm = []
		for expr in X:
			expr_s = np.arctan(expr)
			expr_ns = (expr_s - np.mean(expr_s, axis=0, keepdims=True))/(1e-8 + np.std(expr_s, axis=0, keepdims=True))
			X_norm.append(expr_ns)
		return X_norm
		

def visualise_cellgraph(graph, random_state=42, node_labels=None, show=True, spatial_coords=None, ax=None, pos=None, add_legend=False, label_to_color=None, largest_comp=None):
	"""
	Visualizes a cell graph using NetworkX and Matplotlib.

	Args:
		graph: A scipy.sparse matrix representing the cell graph.
		random_state: An integer seed for reproducibility of the layout algorithm (default: 42).
		node_labels: An optional numpy array of node labels to color-code the nodes.
	"""
	np.random.seed(random_state)

	edges = []
	for i in range(graph.shape[0]):
		for j in graph.indices[graph.indptr[i]:graph.indptr[i+1]]:
			edges.append((i, j))
	# Create a NetworkX graph and add edges
	G = nx.Graph()
	G.add_edges_from(edges)
	G.remove_edges_from(nx.selfloop_edges(G))

	if largest_comp is not None:
		# Identify the largest connected component
		largest_cc = max(nx.connected_components(G), key=len)
		G = G.subgraph(largest_cc).copy()
	if node_labels is not None:
		node_labels = np.array(node_labels)
		node_labels = [node_labels[i] for i in G.nodes]

	if ax is None:
		# Set appropriate figure size for large graphs
		fig, ax = plt.subplots(figsize=(10, 6))
	else:
		fig = ax.get_figure()
	# Use a layout that handles large graphs relatively well
	if pos is None:
		if spatial_coords is None:
			pos = nx.spring_layout(G, seed=random_state, k=0.15, iterations=200)
		else:
			pos = {i: (spatial_coords[i][0], spatial_coords[i][1]) for i in range(graph.shape[0])}
	# Create a color mapper for normalization
	unique_labels = list(set(node_labels))
	if node_labels is not None and label_to_color is None:
		# Choose a colormap (modify as needed)
		cmap = plt.cm.tab20  # Select a colormap from matplotlib.cm
		norm = plt.Normalize(vmin=0, vmax=len(unique_labels) - 1) 
		sm = ScalarMappable(cmap=cmap, norm=norm)
		label_to_color = {label: sm.to_rgba(i) for i, label in enumerate(unique_labels)}

	if node_labels is not None:
		# Draw nodes with colors based on labels and colormap
		node_colors = [label_to_color[label] for label in node_labels]
		nx.draw_networkx_nodes(G, pos, node_size=5, node_color=node_colors, ax=ax)
	else:
		nx.draw_networkx_nodes(G, pos, node_size=5, ax=ax)

	nx.draw_networkx_edges(G, pos, width=0.2, alpha=0.5, edge_color='gray', ax=ax)
	
	if node_labels is not None and add_legend:
		legend_handles = [patches.Patch(color=sm.to_rgba(i), label=label) for i, label in enumerate(unique_labels)]
		ax.legend(handles=legend_handles, ncol=5, loc='upper center', bbox_to_anchor=(1.1, 1), borderaxespad=0.)
	ax.axis('off')
	if show:
		plt.show()
	return fig, ax, pos, label_to_color