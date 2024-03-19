import scanpy as sc
import numpy as np
from .utils import load_cell_data, cellcell_to_features
import os
import pickle
from mainutils.utils import train_test_split, k_fold_split

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
		filename = f"{self.config['DATA_PATH']}/processed_data_{self.config['gtype']}.pkl"

		if os.path.exists(f"{filename}"):		
			self.data_train, self.data_test = self.load_split_data(filename)
		else:
			self.data_train, self.data_test = self.prepare_data(filename)

		# if os.path.exists(f"{filename}"):		
		# 	self.data = self.load_data(filename)
		# else:
		# 	self.data = self.prepare_data(filename)

		self.unique_labels = {'pCR': 1, 'Responder': 1, 'Non-Responder': 0}
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

	def load_split_data(self, filename):
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
			return data['train'], data['test']

	def prepare_data(self, filename):
		"""
		Loads raw cell data, preprocesses it, and saves the processed data to a file.

		Args:
			filename (str): Path to the pickle file where the processed data will be saved.

		Returns:
			dict: A dictionary containing: expressions, enrichments (default None), graphs (default None), labels, and feature names.
		"""
		print('Loading Cell Table')
		cell_table = load_cell_data(self.config['DATA_PATH'],
												self.config['cell_filename'], 
												self.config['response_filename'])

		print('Preparing Expression Data From Cell Table and saving to disk')
		data = cellcell_to_features(cell_table, 
										gmethod=self.config['gmethod'], 
										k=self.config['k'],
										filename=filename)

		print('Split Expression Data and save to disk')
		if self.config['datasplit'] == 'split':
			data_train, data_test = train_test_split(data, 
													test_size=self.config['test_ratio'], 
													random_state=self.config['seed'])

			dataset = {
						'train': data_train,
						'test': data_test
			}

			with open(f"{self.config['DATA_PATH']}/{self.config['gtype']}_processed_split.pkl") as f:
				pickle.dump(dataset, f)
			return data_train, data_test

		print('Split Expression Data and save to disk')
		if self.config['datasplit'] == 'kfold':
			folds = k_fold_split(data, 
									test_size=self.config['test_ratio'], 
									random_state=self.config['seed'])


			with open(f"{self.config['DATA_PATH']}/{self.config['gtype']}_processed_split.pkl") as f:
				pickle.dump(dataset, f)
			return folds, None

