import numpy as np
import pandas as pd
import scanpy as sc
import os
import pickle
from utils.utils import coords_to_graph
from scipy.sparse import csr_matrix
import squidpy as sq
from collections import OrderedDict

##
# Based On Giuseppe's Code

MARKERS = ['Alpha-SMA', 'B7-H4', 'Beta-Catenin', 'CD107a', 'CD11b', 'CD14', 'CD16', 
			'CD163', 'CD20', 'CD27', 'CD3', 'CD31', 'CD366', 'CD38', 'CD4', 'CD44', 
			'CD45', 'CD45RO', 'CD68', 'CD8a', 'Carboplatin', 'Collage-Type_I', 
			'DNA1', 'DNA2', 'E-Cadherin', 'EGFR', 'FOXP3', 'Granzyme-B', 'HLA-DR-DQ-DP', 
			'Ki-67', 'PD-1', 'PD-L1', 'PD-L2', 'Pan-keratin', 'Tbet', 'VEGF', 
			'Vimentin', 'p53']

def binarise(data, thr):
	"""
	Binarise the data to 0/1

	Args:
		data: Anndata object or 2D NumPy array of data to normalise.
		thr: threshold to binary the data.
	Returns:
		Binarised data as per `thr`
	"""
	return data.X>thr if isinstance(data, sc.AnnData) else data>thr

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

	if not isinstance(adata, (np.ndarray, sc.AnnData)):
		raise ValueError("`adata` must be an Anndata object or a 2D NumPy array.")

	# Convert to array efficiently and check for scaling
	data = adata.X if isinstance(adata, sc.AnnData) else np.asarray(adata, dtype=np.float64)
	if np.all(data >= 0) and np.all(data <= 1):
		logging.warning("Data seems already normalized, skipping normalisation")
		return adata

	# Scaling and clipping using robust NaN handling
	q = np.nanquantile(data, q=quantile, axis=0)  # Use np.nanquantile to ignore NaNs
	data /= q[None, :]
	data = np.clip(data, 0, 1)
	# Update Anndata or return array

	if isinstance(adata, sc.AnnData):
		adata.X = data
		return adata
	else:
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
	is_protein_data = data.columns.get_loc('label') > 1
	is_marker = data.columns.isin(MARKERS)
	is_protein_marker = is_protein_data & is_marker

	# Perform filtering and quantile calculation efficiently using broadcasting
	active_genes_few = (binarise(
					data.loc[:,is_protein_marker], thr=low_gene_active) > 0).sum(axis=1)
	active_genes_many = (binarise(
					data.loc[:,is_protein_marker], thr=high_gene_active) < 11).sum(axis=1)
	dna_thr = np.quantile(data[['DNA1', 'DNA2']].sum(axis=1), dna_quantile)
	passed_qc = active_genes_few & active_genes_many & (intensities[['DNA1', 'DNA2']].sum(axis=1) > dna_thr)
	return passed_qc

def load_cell_data(datapath, filename_celldata, filename_biosamples):
	"""
	Loads cell data from CSV files and performs quality control if needed.

	Args:
		datapath (str): Path to the data directory.
		filename_celldata (str): Name of the cell data CSV file.
		filename_biosamples (str): Name of the biosamples CSV file.

	Returns:
		tuple: A tuple containing the loaded cell table and biosamples DataFrames.
	"""
	cell_table = pd.read_csv(f'{datapath}/{filename_celldata}.csv', sep=',')
	if 'qc_pass' not in cell_table.columns:
		qc_pass = quality_control(cell_data)
		cell_table['qc_pass'] = qc_pass
		cell_table.to_csv(f'{datapath}/{filename_celldata}.csv', index=False)
	biosamples = pd.read_csv(f'{datapath}/{filename_biosamples}.csv', sep=',')
	biosamples.drop(['FORCE_TRIAL?_(Y/N)'],axis = 1,inplace = True)        
	return cell_table, biosamples

def celltable_to_anndata(cell_table, biosamples):
	"""
	Creates an AnnData object from cell data and merges metadata.

	Args:
		cell_table (pandas.DataFrame): Cell data DataFrame.
		biosamples (pandas.DataFrame): Biosamples DataFrame.

	Returns:
		anndata.AnnData: An AnnData object containing the processed data.
	"""
	if 'cell_meta_cluster' in cell_table:
		cell_table = cell_table[cell_table['cell_meta_cluster']!='Unassigned']
	adata = sc.AnnData(cell_table.loc[:,cell_table.columns.isin(MARKERS)], obsm={"spatial": cell_table[['centroid-0', 'centroid-1']].values})
	try:
		adata.obs['Pixie'] = pd.Categorical(cell_table.cell_meta_cluster.values.astype(str))
	except:
		print('cell type label not present')

	adata.obs['acquisition_ID'] = cell_table.fov.values
	adata.obs['Leap_ID'] = adata.obs.acquisition_ID.str.split('_',n = 1).str[0].str.upper()
	adata.obs['Leap_ID'] = adata.obs.Leap_ID.str[:7]#leap_ID should be Leap123, anything more is stripped
	adata.obs = adata.obs.reset_index().merge(biosamples,left_on='Leap_ID',right_on= 'LEAP_ID').drop(['LEAP_ID'],axis = 1).set_index('index')
	
	adata.obs['qc_pass'] = cell_table['qc_pass'].values
	adata = adata[~((adata.obs.Response == 'Responder')&(adata.obs['SAMPLE_TYPE_(CORE/RESECTION)']=='RESECTION'))]#remove cases of resection of responders

	# get fovs having more than 1000 cells
	fovs = adata.obs.acquisition_ID.value_counts()[adata.obs.acquisition_ID.value_counts()>=1000].index
	adata = adata[adata.obs.acquisition_ID.isin(fovs)]
	adata.raw = adata#raw data are unfiltered and unnormalised

	adata.X[np.isnan(adata.X)] =0#the nan comes when a  segmented file does not have the corresponding channel tiff file. That happened for the Carboplatin on a release that dates to Jan 24. On a new full process of data, check that this is not required anymore

	#Normalise each channel independently by quantile
	adata = normalise(adata, quantile=0.95)    
	return adata  


def filter_data(adata, qc_pass=False, use_core=True):
	"""
	Filters the AnnData object based on user-defined criteria.

	Args:
		adata (anndata.AnnData): The AnnData object containing the data.
		qc_pass (bool, optional): If True, filter to include only high-quality cells based on the 'qc_pass' label. Defaults to False.
		use_core (bool, optional): If True, filter to include only core biopsies based on the 'SAMPLE_TYPE_(CORE/RESECTION)' label. Defaults to True.

	Returns:
		anndata.AnnData: The filtered AnnData object.
	"""

	if use_core:
		adata = adata[adata.obs['SAMPLE_TYPE_(CORE/RESECTION)']=='CORE']
	if qc_pass:
		adata = adata[adata.obs['qc_pass']]
	return adata

def anndata_to_datatensor(adata):
	"""
	Converts AnnData object to data tensor with mean expression per "acquisition_ID" and "Pixie" combination.

	Args:
		adata: AnnData object containing gene expression data.

	Returns:
		numpy.ndarray: Data tensor with shape (n_unique_acquisition_IDs, n_unique_celltypes, n_proteins).
	"""
	adata = filter_data(adata)
	n_proteins = adata.X.shape[1]
	# Group by 'Leap_ID' and 'Pixie' and calculate the mean for each group
	grouped_data = adata.obs.groupby(['acquisition_ID', 'Pixie'])
	celltype_idx = {celltype:j for j, celltype in enumerate(adata.obs['Pixie'].unique())}
	acqn_idx = {acqid: i for i, acqid in enumerate(adata.obs['acquisition_ID'].unique())}

	data_tensor = np.zeros((len(acqn_idx), len(celltype_idx), n_proteins), dtype=np.float64)
	acqnidx_label = OrderedDict()

	for (acqn, celltype), idx in grouped_data.groups.items():
		data_tensor[acqn_idx[acqn], celltype_idx[celltype],:] = adata[idx].X.mean(0)
		if acqn in acqnidx_label:
			continue
		acqnidx_label[acqn] = adata.obs.Response[idx].unique()

	labels = [label.item() for _, label in acqnidx_label.items()]
	return data_tensor, labels


def celltype_to_features(adata, filename='./data.pkl', cell_radius=20, cell_n_thr=50):
	"""
	Extracts features based on cell types and spatial relationships from an AnnData object.

	Args:
		adata (anndata.AnnData): AnnData object containing spatial transcriptomics data.
		filename (str, optional): Filename to save processed data. Defaults to './data.pkl'.
		cell_radius (int, optional): Radius for identifying spatial neighbors. Defaults to 20.
		cell_n_thr (int, optional): Minimum number of cells for a cell type to be considered. Defaults to 50.

	Returns:
		tuple: A tuple containing expressions, enrichments, graphs, labels, and markers.
	"""
	acqns = adata.obs.acquisition_ID.unique()

	pixies = np.array(adata.obs.Pixie.cat.categories)
	expression = pd.DataFrame(np.zeros((len(pixies), len(MARKERS))), index=pixies, columns=MARKERS)

	enrichments, expressions, graphs, labels = [], [], [], []
	for idx in acqns:
		sub_adata = adata[adata.obs.acquisition_ID==idx]
		celltypes = sub_adata.obs.Pixie.value_counts()[sub_adata.obs.Pixie.value_counts()>cell_n_thr].index.values

		sq.gr.spatial_neighbors(sub_adata,coord_type='grid',n_neighs=6,radius = (0,cell_radius))
		sq.gr.nhood_enrichment(sub_adata, cluster_key='Pixie')
		sq.gr.interaction_matrix(sub_adata, cluster_key='Pixie')

		a = np.array(sub_adata.obs['Pixie'].cat.categories)
		enrichment = sub_adata.uns['Pixie_nhood_enrichment']['zscore']
		enrichment = pd.DataFrame(enrichment,index=a,columns=a)
		enrichment = enrichment.loc[celltypes, celltypes]
		enrichments.append(enrichment)

		contact = sub_adata.uns['Pixie_interactions']/sub_adata.uns['Pixie_interactions'].sum()
		contact = pd.DataFrame(contact,index=a,columns=a)
		contact = contact.loc[celltypes, celltypes]

		graphs.append(csr_matrix(contact))
		label = set(sub_adata.obs.Response.values)
		expression.loc[a,:] = np.array([sub_adata[loc].X.mean(0) for _,loc in sub_adata.obs.groupby(['Pixie']).groups.items()])

		expressions.append(expression)
		expression = pd.DataFrame(np.zeros((len(pixies), len(MARKERS))), index=pixies, columns=MARKERS)
		if len(label) != 1:
			assert 0, f"Acquistion {idx} has non unique labels"
		labels.append(label.pop())
	dataset = { 
				'expressions': expressions,
				'enrichments': enrichments,
				'graphs': graphs,
				'labels': labels,
				'markers': MARKERS
		}

	with open(f"{filename}", 'wb') as f:
		pickle.dump(dataset, f)			
	return dataset

def cellcell_to_features(adata, min_cells=10, gmethod='knn', k=7, filename='./data.pkl'):
	"""
	Extracts features based on interactions between individual cells within each acquisition.

	Args:
		adata (anndata.AnnData): AnnData object containing spatial transcriptomics data.
		min_cells (int, optional): Minimum number of cells required in an acquisition for processing. Defaults to 10.
		gmethod (str, optional): Method for constructing the adjacency matrix (e.g., 'knn'). Defaults to 'knn'.
		k (int, optional): Number of nearest neighbors for the kNN method. Defaults to 7.
		filename (str, optional): Filename to save processed data. Defaults to './data.pkl'.

	Returns:
		tuple: A tuple containing expressions, graphs, labels, and markers.
	"""
	unique_acqns = adata.obs['acquisition_ID'].unique()
	celltype_idx = {celltype:j for j, celltype in enumerate(adata.obs['Pixie'].unique())}
	acqn_idx = {acqid: i for i, acqid in enumerate(adata.obs['acquisition_ID'].unique())}
	expressions, graphs, labels = [], [], []
	for acq in unique_acqns:
		sub_adata = adata[adata.obs.acquisition_ID==acq]
		coords = sub_adata.obsm['spatial']
		if len(coords)<min_cells:
			continue
		expressions.append(sub_adata.X)
		graph = coords_to_graph(coords, gmethod='knn', radius=k)
		graphs.append(graph)
		label = set(sub_adata.obs.Response.values)
		if len(label) != 1:
			assert 0, f"Acquistion {idx} has non unique labels"
		labels.append(label.pop())
	dataset = {
				'expressions': expressions,
				'graphs': graphs,
				'labels': labels,
				'enrichments': None,
				'markers': MARKERS
		}

	with open(f"{filename}", 'wb') as f:
		pickle.dump(dataset, f)			
	return dataset

