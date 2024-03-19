import numpy as np
import pandas as pd
import os
import pickle
from mainutils.utils import coords_to_graph
from scipy.sparse import csr_matrix
import squidpy as sq
from collections import OrderedDict

from functools import partial
from tqdm import tqdm
from joblib import Parallel, delayed
##
# Based On Giuseppe's Code

ALLMARKERS = ['Alpha-SMA', 'B7-H4', 'Beta-Catenin', 'CD107a', 'CD11b', 'CD14', 'CD16', 
			'CD163', 'CD20', 'CD27', 'CD3', 'CD31', 'CD366', 'CD38', 'CD4', 'CD44', 
			'CD45', 'CD45RO', 'CD68', 'CD8a', 'Carboplatin', 'Collage-Type_I', 
			'DNA1', 'DNA2', 'E-Cadherin', 'EGFR', 'FOXP3', 'Granzyme-B', 'HLA-DR-DQ-DP', 
			'Ki-67', 'PD-1', 'PD-L1', 'PD-L2', 'Pan-keratin', 'Tbet', 'VEGF', 
			'Vimentin', 'p53']

EXCLUDE_MARKERS = ['Carboplatin', 'Collage-Type_I', 'DNA1', 'DNA2', 'p53', 'VEGF', 
				'EGFR', 'Ki-67', 'PD-1', 'PD-L1', 'PD-L2']

MARKERS = list(filter(lambda x: x not in EXCLUDE_MARKERS, ALLMARKERS))

def binarise(data, thr):
	"""
	Binarise the data to 0/1

	Args:
		data: Anndata object or 2D NumPy array of data to normalise.
		thr: threshold to binary the data.
	Returns:
		Binarised data as per `thr`
	"""
	return data>thr

def normalise(adata, quantile=0.75):
	"""
	Normalizes `adata` using quantile normalisation, and ignoring NaNs.

	Args:
		adata: Anndata object or 2D NumPy array of data to normalise.
		quantile (float, optional): Quantile used for normalisation (default: 0.75).

	Returns:
		Normalised Anndata object or NumPy array, depending on the input type.

	Raises:
		ValueError: If `adata` is not an Anndata object or a 2D NumPy array.
	"""

	if not isinstance(adata, np.ndarray):
		raise ValueError("`adata` must be a 2D NumPy array.")

	# Convert to array efficiently and check for scaling
	if np.all(data >= 0) and np.all(data <= 1):
		logging.warning("Data seems already normalized, skipping normalisation")
		return adata

	# Scaling and clipping using robust NaN handling
	q = np.nanquantile(data, q=quantile, axis=0)  # Use np.nanquantile to ignore NaNs
	data /= q[None, :]
	data = np.clip(data, 0, 1)
	return data

def quality_control(data, low_gene_active=0.2, high_gene_active=0.5, dna_quantile=0.05):
	"""
	Performs quality control filtering on intensity data, ensuring efficiency and readability.

	Args:
		data (pd.DataFrame): DataFrame containing intensity data.
		low_gene_active (float, optional): Threshold for minimum active genes per cell (default: 0.2).
		high_gene_active (float, optional): Threshold for maximum active genes per cell (default: 0.5).
		dna_quantile (float, optional): Quantile for DNA content filtering (default: 0.05).

	Returns:
		pd.Series: Boolean Series indicating cells that pass quality control.
	"""

	if 'pass_qc' in data.columns:
		return data['pass_qc']

	# Create efficient boolean masks for filtering
	is_marker = data.columns.isin(MARKERS)

	# Perform filtering and quantile calculation efficiently using broadcasting
	active_genes_few = (binarise(
					data.loc[:,is_marker], thr=low_gene_active).sum(axis=1)>0)
	active_genes_many = (binarise(
					data.loc[:,is_marker], thr=high_gene_active).sum(axis=1)<11)
	dna_thr = np.quantile(data[['DNA1', 'DNA2']].sum(axis=1), dna_quantile)
	passed_qc = active_genes_few & active_genes_many & (data[['DNA1', 'DNA2']].sum(axis=1) > dna_thr)
	return passed_qc

def load_cell_data(datapath, filename_celldata, filename_biosamples):
	"""
	Loads cell data from CSV files and performs quality control if needed.

	Args:
		datapath (str): Path to the data directory.
		filename_celldata (str): Name of the cell data CSV file.
		filename_biosamples (str): Name of the biosamples CSV file.

	Returns:
		pd.DataFrame: A combined dataframe containing cell table and biosamples DataFrames.
	"""
	cell_table = pd.read_csv(f'{datapath}/{filename_celldata}.csv', sep=',')
	if 'qc_pass' not in cell_table.columns:
		qc_pass = quality_control(cell_data)
		cell_table['qc_pass'] = qc_pass
		cell_table.to_csv(f'{datapath}/{filename_celldata}.csv', index=False)
	biosamples = pd.read_csv(f'{datapath}/{filename_biosamples}.csv', sep=',')
	biosamples.drop(['FORCE_TRIAL?_(Y/N)'],axis = 1,inplace = True)        

	if 'cell_meta_cluster' in cell_table:
		cell_table = cell_table[cell_table['cell_meta_cluster']!='Unassigned']

	cell_table['LEAP_ID'] = cell_table.fov.str.split('_', n=1).str[0].str.upper()
	cell_table['LEAP_ID'] = cell_table.LEAP_ID.str[:7]#leap_ID should be Leap123, anything more is stripped

	cell_table = cell_table.reset_index().merge(biosamples, left_on='LEAP_ID', right_on= 'LEAP_ID').drop(['LEAP_ID'], axis = 1).set_index('index')

	# get fovs having more than 1000 cells
	fovs = cell_table.fov.value_counts()[cell_table.fov.value_counts()>=1000].index
	cell_table = cell_table[cell_table.fov.isin(fovs)]
	cell_table[MARKERS] = cell_table[MARKERS].fillna(0)
	return cell_table  


def filter_data(cell_table, qc_pass=False, use_core=True):
	"""
	Filters the AnnData object based on user-defined criteria.

	Args:
		cell_table (pd.DataFrame): The pandas object containing the data.
		qc_pass (bool, optional): If True, filter to include only high-quality cells based on the 'qc_pass' label. Defaults to False.
		use_core (bool, optional): If True, filter to include only core biopsies based on the 'SAMPLE_TYPE_(CORE/RESECTION)' label. Defaults to True.

	Returns:
		pd.DataFrame: The filtered pd frame.
	"""

	if use_core:
		cell_table = cell_table[cell_table['SAMPLE_TYPE_(CORE/RESECTION)']=='CORE']
	if qc_pass:
		cell_table = cell_table[cell_table['qc_pass']]
	return cell_table

def process_roi(roi, cell_table, min_cells, leap_to_patient, k=6):
	roi_cells = cell_table[cell_table.fov == roi]
	if len(roi_cells) < min_cells:
		return None

	coords = roi_cells[['centroid-0', 'centroid-1']].values
	expressions = roi_cells[MARKERS].values
	cell_labels = roi_cells.cell_meta_cluster.values
	graph = coords_to_graph(coords, gmethod='knn', radius=k)
	label = set(roi_cells.Response.values)
	if len(label) != 1:
		assert 0, f"Acquisition {roi} has non-unique labels"
	label = label.pop()
	patient = leap_to_patient[roi.split('_')[0]]

	return expressions, graph, label, patient, cell_labels


def cellcell_to_features(cell_table, min_cells=10, gmethod='knn', k=6, filename='./data.pkl', num_cores=8):
	"""
	Extracts features based on interactions between individual cells within each acquisition.

	Args:
		adata (anndata.AnnData): AnnData object containing spatial transcriptomics data.
		min_cells (int, optional): Minimum number of cells required in an acquisition for processing. Defaults to 10.
		gmethod (str, optional): Method for constructing the adjacency matrix (e.g., 'knn'). Defaults to 'knn'.
		k (int, optional): Number of nearest neighbors for the kNN method. Defaults to 7.
		filename (str, optional): Filename to save processed data. Defaults to './data.pkl'.

	Returns:
		dict: A dictionary containing expressions, graphs, labels, markers and patientid.
	"""
	unique_rois = cell_table.fov.unique()
	unique_leaps = set(roi.split('_')[0] for roi in unique_rois)
	leap_to_patient = {leap: f"patient_{i+1}" for i, leap in enumerate(unique_leaps)}
	celltypes = np.array(cell_table.cell_meta_cluster.unique())

	dataset = {
		'expressions': [],
		'graphs': [],
		'labels': [],
		'patient': [],
		'cell_labels': [],
		'markers': MARKERS,
		'celltypes': celltypes
		}

	results = Parallel(n_jobs=num_cores)(
			delayed(process_roi)(roi, cell_table, min_cells, leap_to_patient, k)
			for roi in tqdm(unique_rois, desc="Processing ROIs")
	)

	for result in results:
		if result is not None:
			expressions, graph, label, patient, cell_labels = result
			dataset['expressions'].append(expressions)
			dataset['graphs'].append(graph)
			dataset['labels'].append(label)
			dataset['patient'].append(patient)
			dataset['cell_labels'].append(cell_labels)

	with open(filename, 'wb') as f:
		pickle.dump(dataset, f)

	return dataset
