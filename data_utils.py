import numpy as np
import pandas as pd
import scanpy as sc
import os


##
# Based On Giuseppe's Code

MARKERS = ['CD38', 'CD14', 'Tbet', 'CD16', 'CD163',
			'Pan-keratin', 'CD11b', 'CD107a', 'CD45', 'CD44', 'CD366',
			'FOXP3', 'CD4', 'E-Cadherin', 'CD68', 'HLA-DR-DQ-DP', 'CD20',
			'CD8a', 'Beta-Catenin', 'B7-H4', 'Granzyme-B',
			'CD3', 'CD27', 'CD45RO',
			'Alpha-SMA', 'Vimentin', 'CD31' ]

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
	cell_table = pd.read_csv(f'{datapath}/{filename_celldata}.csv', sep=',')
	if 'qc_pass' not in cell_table.columns:
		qc_pass = quality_control(cell_data)
		cell_table['qc_pass'] = qc_pass
		cell_table.to_csv(f'{datapath}/{filename_celldata}.csv', index=False)
	biosamples = pd.read_csv(f'{datapath}/{filename_biosamples}.csv', sep=',')
	biosamples.drop(['FORCE_TRIAL?_(Y/N)'],axis = 1,inplace = True)        
	return cell_table, biosamples

def celltable_to_anndata(cell_table, biosamples):
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

def anndata_to_datatensor(adata):
	"""
	Converts AnnData object to data tensor with mean expression per "acquisition_ID" and "Pixie" combination.

	Args:
		adata: AnnData object containing gene expression data.

	Returns:
		numpy.ndarray: Data tensor with shape (n_unique_acquisition_IDs, n_unique_celltypes, n_proteins).
	"""
	n_proteins = adata.X.shape[1]
	# Group by 'Leap_ID' and 'Pixie' and calculate the mean for each group
	grouped_data = adata.obs.groupby(['acquisition_ID', 'Pixie'])
	celltype_idx = {celltype:j for j, celltype in enumerate(adata.obs['Pixie'].unique())}
	acqid_idx = {acqid: i for i, acqid in enumerate(adata.obs['acquisition_ID'].unique())}

	acq_to_response = {}
	for acqid, label in zip(adata.obs['acquisition_ID'], adata.obs['Response']):
		if acqid in acq_to_response:
			continue
		else:
			acq_to_response[acqid] = label

	data_tensor = np.zeros((len(acqid_idx), len(celltype_idx), n_proteins), dtype=np.float64)
	labels = []
	acq_prev = 'refewa'
	for i, (acqn, celltype), idx in enumerate(grouped_data.groups.items()):
		if i == 0:
			acq_prev = acqn
		if adata.obs.qc_pass[idx].all() == True:
			data_tensor[acqid_idx[acqn], celltype_idx[celltype],:] = adata[idx].X.mean(0)
			if acqn == acq_prev:
				continue
			acq_prev = acqn
			labels.append(acq_to_response[acqn])
	return data_tensor, acq_to_response


from collections import OrderedDict

def anndata_to_datatensor(adata):
	"""
	Converts AnnData object to data tensor with mean expression per "acquisition_ID" and "Pixie" combination.

	Args:
		adata: AnnData object containing gene expression data.

	Returns:
		numpy.ndarray: Data tensor with shape (n_unique_acquisition_IDs, n_unique_celltypes, n_proteins).
	"""
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