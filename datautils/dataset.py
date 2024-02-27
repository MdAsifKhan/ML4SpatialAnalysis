import scanpy as sc
import numpy as np
from .utils import celltable_to_anndata, load_cell_data, cellcell_to_features, celltype_to_features
import os
import pickle

class SpatialCellToFeatures:
	"""
	This class loads and prepares spatial gene expression data for further analysis.

	Attributes:
		config (dict): Configuration dictionary containing data and processing parameters.
		expressions (np.ndarray): Array containing gene expression data.
		enrichments (np.ndarray, optional): Array containing enrichment features (if applicable). Defaults to None.
		graphs (list, optional): List of graphs representing spatial relationships (if applicable). Defaults to None.
		labels (list): List of ground truth labels.
		feature_names (list): List of feature names.
		label_vec (np.ndarray): One-hot encoded labels.
	"""
	def __init__(self, config):
		"""
		Initializes the SpatialCellToFeatures object.

		Args:
			config (dict): Configuration dictionary containing data and processing parameters.
		"""

		self.config = config
		if self.config['gtype'] == 'cellcell':
			filename = f"{self.config['DATA_PATH']}/processed_data_{self.config['gtype']}.pkl"
		else:
			filename = f"{self.config['DATA_PATH']}/processed_data_{self.config['gtype']}_cellr{self.config['cell_radius']}_cellt{self.config['cell_n_thr']}.pkl"

		if os.path.exists(f"{filename}"):		
			self.data = self.load_data(filename)
		else:
			self.data = self.prepare_data(filename)

		self.unique_labels = {el:i for i, el in enumerate(list(set(self.data['labels'])))}
		self.data['labels'] = np.asarray([self.unique_labels[label] for label in self.data['labels']])


	def load_data(self, filename):
		"""
		Loads pre-processed data from a pickle file.

		Args:
			filename (str): Path to the pickle file containing the data.

		Returns:
			dict: A dictionary containing: expressions, enrichments (default None), graphs (default None), labels, and feature names.
		"""
		print('Loading Expression Data From File')
		with open(filename, 'rb') as f:
			data = pickle.load(f)
			return data

	def prepare_data(self, filename):
		"""
		Loads raw cell data, preprocesses it, and saves the processed data to a file.

		Args:
			filename (str): Path to the pickle file where the processed data will be saved.

		Returns:
			dict: A dictionary containing: expressions, enrichments (default None), graphs (default None), labels, and feature names.
		"""
		print('Preparing Expression Data From Cell Table')
		cell_table, biosamples = load_cell_data(self.config['DATA_PATH'],
												self.config['cell_filename'], 
												self.config['response_filename'])

		adata = celltable_to_anndata(cell_table, biosamples)
		if self.config['gtype'] == 'cellcell':
			data = cellcell_to_features(adata, 
											gmethod=self.config['gmethod'], 
											k=self.config['k'],
											filename=filename)

			return dataset
		if self.config['gtype'] == 'celltype':
			data = celltype_to_features(adata, 
												cell_radius=self.config['cell_radius'],
												cell_n_thr=self.config['cell_n_thr'],
												filename=filename)
			return data
		else:
			assert 0, f"{self.config['gtype']} Not Implemented"	

