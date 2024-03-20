import pandas as pd
import matplotlib.pyplot as plt

datapath = '/Users/asifkhan/workspace/SpatialCellAnalysis/ML4SpatialAnalysis/data'
cell_filename = 'cell_table_size_normalized_cell_labels_corrected_labels'  # Filename containing cell table data
response_filename = 'processed_response'

ALLMARKERS = ['Alpha-SMA', 'B7-H4', 'Beta-Catenin', 'CD107a', 'CD11b', 'CD14', 'CD16', 
			'CD163', 'CD20', 'CD27', 'CD3', 'CD31', 'CD366', 'CD38', 'CD4', 'CD44', 
			'CD45', 'CD45RO', 'CD68', 'CD8a', 'Carboplatin', 'Collage-Type_I', 
			'DNA1', 'DNA2', 'E-Cadherin', 'EGFR', 'FOXP3', 'Granzyme-B', 'HLA-DR-DQ-DP', 
			'Ki-67', 'PD-1', 'PD-L1', 'PD-L2', 'Pan-keratin', 'Tbet', 'VEGF', 
			'Vimentin', 'p53']

EXCLUDE_MARKERS = ['Carboplatin', 'Collage-Type_I', 'DNA1', 'DNA2', 'p53', 'VEGF', 
				'EGFR', 'Ki-67', 'PD-1', 'PD-L1', 'PD-L2']

MARKERS = list(filter(lambda x: x not in EXCLUDE_MARKERS, ALLMARKERS))

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


cell_table, biosamples = load_cell_data(datapath, cell_filename, response_filename)
