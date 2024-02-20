import scanpy as sc
import numpy as np
from .utils import celltable_to_anndata, load_cell_data, anndata_to_datatensor, anndata_to_features
import os
from utils.utils import graph_feature_vector
import pickle


class SpatialCellToFeatures:
	def __init__(self, config):
		self.config = config
		if os.path.exists(f"{self.config['DATA_PATH']}/processed_data_{self.config['gtype']}.pkl"):		
			self.expressions, self.graphs, self.labels = self.load_data()
		else:
			self.expressions, self.graphs, self.labels = self.prepare_data()

		unique_labels = {el:i for i, el in enumerate(list(set(self.labels)))}
		self.label_vec = np.asarray([unique_labels[label] for label in self.labels])
		if self.config['use_graph']:
			self.graph_features = self.graph_features()

	def load_data(self):
		with open(f"{self.config['DATA_PATH']}/processed_data_{self.config['gtype']}.pkl", 'rb') as f:
			dataset = pickle.load(f)
			return dataset['expressions'], dataset['graphs'], dataset['labels']

	def prepare_data(self):
		cell_table, biosamples = load_cell_data(self.config['DATA_PATH'],
												self.config['cell_filename'], 
												self.config['response_filename'])

		adata = celltable_to_anndata(cell_table, biosamples)
		expressions, graphs, labels = anndata_to_features(adata, 
											gtype=self.config['gtype'],
											datapath=self.config['DATA_PATH'])
		return expressions, graphs, labels


	def graph_features(self):
		gfeature = [graph_feature_vector(graph, self.config['gcriterion']) for graph in self.graphs]
		return np.array(gfeature)

	def exprn_tocelltype_featurisation(self):
		data_mat = np.array(self.expressions)
		import pdb
		pdb.set_trace()
		if self.config['fcriterion'] == 'avgcelltype':
			return np.mean(self.expressions, axis=1)
		elif self.config['fcriterion'] == 'flatcelltype':
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
			e_features = self.exprn_tocelltype_featurisation()
		else:
			assert 0, f"{self.config['gtype']} Expression Features are invalid"
		if self.config['use_graph']:
			features = np.cat([e_features, self.graph_features], axis=1)
			return features
		return e_features

