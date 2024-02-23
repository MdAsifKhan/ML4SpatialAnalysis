from sklearn.linear_model import LogisticRegression
import pickle
from sklearn.model_selection import KFold, train_test_split
from utils.utils import compute_scores_train, compute_scores_test
import numpy as np
from utils.utils import graph_feature_vector, coords_to_graph
#from factory import GCN

class ModelTrainer:
	def __init__(self, config, feature_names=None):
		self.config = config
		self.feature_names = feature_names
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
			#self.classifier = GCN()
			pass
		else:
			assert 0,f"Classifer {self.config['name']} is not implemented"

	def fit(self, X, y, graphs=None):
		if self.config['name'] == 'gcn':
			self.classifier.fit(X, y, graphs)
			return
		self.classifier.fit(X, y)

	def predict(self, X, graphs=None):
		if self.config['name'] == 'gcn':
			return self.classifier.predict(X, graphs)
		return self.classifier.predict(X)

	def predict_proba(self, X, graphs=None):
		if self.config['name'] == 'gcn':
			return self.classifier.predict_proba(X, graphs)
		return self.classifier.predict_proba(X)

	def save_model(self, filename):
		filename = f"{self.config['MODEL_PATH']}/{self.config['name']}_{filename}.pkl"
		out = {
				'model': self.classifier,
				'feature_names': self.feature_names
				}
		with open(filename, 'wb') as f:
			pickle.dump(out, f)

	def load_model(self, filename):
		filename = f"{self.config['MODEL_PATH']}/{self.config['name']}_{filename}.pkl"
		with open(filename, 'wb') as f:
			load = pickle.load(f)
			self.classifier = load['model']
			self.feature_names = load['feature_names']


	def graph_features(self, graphs):
		gfeature = [graph_feature_vector(graph, self.config['gcriterion'], self.config['gf_dim']) for graph in graphs]
		return np.array(gfeature)


	def celltype_featurisation(self, X):
		if self.config['fcriterion'] == 'avgcelltype':
			return [expr.mean(axis=0) for expr in X]
		elif self.config['fcriterion'] == 'flatcelltype':
			data_mat = np.array(X)
			n_samples, n_celltype, n_proteins = data_mat.shape
			return data_mat.reshape(n_samples, n_celltype*n_proteins)
		else:
			assert 0,f"{self.config['fcriterion']} Not Implemented"

	def cellcell_to_featurisation(self, X):
		data_mat = [expr.mean(axis=0) for expr in X]
		return np.asarray(data_mat)

	def featurisation(self, X, gtype='cellcell'):
		if gtype == 'cellcell':
			e_features = self.cellcell_to_featurisation(X)
		elif gtype == 'celltype':
			e_features = self.celltype_featurisation(X)
		else:
			assert 0, f"{self.config['gtype']} Expression Features are invalid"
		return e_features

	def optimise(self, X, y, filename, graphs=None, gtype=None):
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
			self.save_model(filename)

		elif self.config['eval'] == 'kfold':
			print(f"Fitting the {self.config['name']} Model")
			kf = KFold(n_splits=self.config['folds'], shuffle=True, random_state=self.config['seed'])

			all_models = []
			y_train, y_test, y_pred_train, y_pred_test = [], [], [], []
			for i, (train_idx, test_idx) in enumerate(kf.split(X)):
				X_train_i, X_test_i = X[train_idx], X[test_idx]
				y_train_i, y_test_i = y[train_idx], y[test_idx]
				self.fit(X_train_i, y_train_i)

				y_train = np.concatenate([y_train, y_train_i])
				y_test = np.concatenate([y_test, y_test_i])

				y_pred_train = np.concatenate([y_pred_train, self.predict(X_train_i)])
				y_pred_test = np.concatenate([y_pred_test, self.predict(X_test_i)])

				self.save_model(f"{filename}_fold_{i+1}")

			print(f"Evaluating the {self.config['name']} Model")
			metrics = compute_scores_train(y_train, y_pred_train, y_test, y_pred_test)

		else:
			assert 0, f"{self.config['eval']} Evaluation not implemented"

		return metrics

	def test(self, X, y, graphs=None):
		if self.config['use_graph_features']:
			print(f"Computing Graph Features criterion {self.config['gcriterion']}")
			graph_features = self.graph_features(graphs)
			X = np.concatenate([X, graph_features], axis=1)
		y_pred = self.predict(X)
		return compute_scores_test(y, y_pred)

