import scanpy as sc
import numpy as np
from .utils import celltable_to_anndata, load_cell_data, cellcell_to_features, celltype_to_features
import os
from utils.utils import graph_feature_vector, coords_to_graph
import pickle
from scipy.sparse import csr_matrix

class SpatialCellToFeatures:
	def __init__(self, config):
		self.config = config
		if self.config['gtype'] == 'cellcell':
			filename = f"{self.config['DATA_PATH']}/processed_data_{self.config['gtype']}.pkl"
		else:
			filename = f"{self.config['DATA_PATH']}/processed_data_{self.config['gtype']}_cellr{self.config['cell_radius']}_cellt{self.config['cell_n_thr']}.pkl"

		if os.path.exists(f"{filename}"):		
			self.expressions, self.enrichments, self.graphs, self.labels, self.feature_names = self.load_data(filename)
		else:
			self.expressions, self.enrichments, self.graphs, self.labels, self.feature_names = self.prepare_data(filename)

		unique_labels = {el:i for i, el in enumerate(list(set(self.labels)))}
		self.label_vec = np.asarray([unique_labels[label] for label in self.labels])

		if self.config['use_graph']:
			print('Preparing Graph Features')
			self.graph_features = self.graph_features()

	def load_data(self, filename):
		print('Loading Expression Data From File')
		with open(filename, 'rb') as f:
			dataset = pickle.load(f)
			return dataset['expressions'], dataset['enrichments'], dataset['graphs'], dataset['labels'], dataset['markers']

	def prepare_data(self, filename):
		print('Preparing Expression Data From Cell Table')
		cell_table, biosamples = load_cell_data(self.config['DATA_PATH'],
												self.config['cell_filename'], 
												self.config['response_filename'])

		adata = celltable_to_anndata(cell_table, biosamples)
		if self.config['gtype'] == 'cellcell':
			expressions, graphs, labels, markers = cellcell_to_features(adata, 
												filename=filename)
			enrichments = None
			return expressions, enrichments, graphs, labels, markers
		if self.config['gtype'] == 'celltype':
			expressions, enrichments, graphs, labels, markers = celltype_to_features(adata, 
												cell_radius=self.config['cell_radius'],
												cell_n_thr=self.config['cell_n_thr'],
												filename=filename)
			return expressions, enrichments, graphs, labels, markers
		else:
			assert 0, f"{self.config['gtype']} Not Implemented"	

	def graph_features(self):
		if self.config['gtype'] == 'cellcell':
			graphs = [coords_to_graph(coords, method=self.config['method'], radius=self.config['k']) for coords in self.graphs]
			gfeature = [graph_feature_vector(graph, self.config['gcriterion'], self.config['gf_dim']) for graph in graphs]
		else:
			gfeature = [graph_feature_vector(csr_matrix(graph), self.config['gcriterion'], self.config['gf_dim']) for graph in self.graphs]
		return np.array(gfeature)

	def celltype_featurisation(self):
		if self.config['fcriterion'] == 'avgcelltype':
			return [expr.mean(axis=0) for expr in self.expressions]
		elif self.config['fcriterion'] == 'flatcelltype':
			data_mat = np.array(self.expressions)
			n_samples, n_celltype, n_proteins = data_mat.shape
			return data_mat.reshape(n_samples, n_celltype*n_proteins)
		else:
			assert 0,f"{self.config['fcriterion']} Not Implemented"

	def cellcell_to_featurisation(self):
		data_mat = [expr.mean(axis=0) for expr in self.expressions]
		return np.asarray(data_mat)

	def featurisation(self):
		if self.config['gtype'] == 'cellcell':
			e_features = self.cellcell_to_featurisation()
		elif self.config['gtype'] == 'celltype':
			e_features = self.celltype_featurisation()
		else:
			assert 0, f"{self.config['gtype']} Expression Features are invalid"
		if self.config['use_graph']:
			features = np.concatenate([e_features, self.graph_features], axis=1)
			gfeatures = [f"Graphcoeff_{i}" for i in range(self.graph_features.shape[1])] 
			self.feature_names += gfeatures
			return features
		return e_features

