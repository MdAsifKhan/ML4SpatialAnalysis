import scanpy as sc
import numpy as np
from .utils import celltable_to_anndata, load_cell_data, anndata_to_datatensor

class SpatialCellToFeatures:
	def __init__(self, config):
		self.config = config
		self.adata = None
		self.load_data()
		self.data_mat, self.labels = self.proteinfeatures()
		unique_labels = {el:i for i, el in enumerate(list(set(self.labels)))}
		self.label_vec = np.asarray([unique_labels[label] for label in self.labels])
		if self.config['use_graph']:
			self.graph_features = self.graph_features()

	def load_data(self):
		cell_table, biosamples = load_cell_data(self.config['DATA_PATH'],
												self.config['cell_filename'], 
												self.config['response_filename'])

		self.adata = celltable_to_anndata(cell_table, biosamples)

	def proteinfeatures(self):
		return anndata_to_datatensor(self.adata)

	def connectivity_matrix(self, qc_pass=False):
		if qc_pass:
			filter_cells = self.adata.obs.qc_pass
		coords = self.adata.obsm['spatial'][filter_cells]
		from sklearn.neighbors import KNeighborsTransformer
		transformer = KNeighborsTransformer(n_neighbors=self.config['knn'],
												mode='distance')

		graph = transformer.fit_transform(coords)
		return graph

	def graph_features(self):
		graph = self.connectivity_matrix()
		from utils import graph_feature_vector
		return graph_feature_vector(graph)


	def featurisation(self):
		if self.config['fcriterion'] == 'avgcelltype':
			return self.data_mat.mean(axis=1)
		elif self.config['fcriterion'] == 'flatcelltype':
			n_samples, n_celltype, n_proteins = self.data_mat.shape
			return self.data_mat.reshape(n_samples, n_celltype*n_proteins)
		else:
			assert 0,f"{self.config['fcriterion']} Not Implemented"