import numpy as np
import pandas as pd
import pickle
import sys
import squidpy as sq
path0 = '../../'
sys.path.append(path0+'phenotyping/')
from phenotyping_utils import generate_anndata_from_ark_analysis

def neighbours(fovs):
    store = {}    
    for ID in  set(fovs):
        sub_adata = adata[adata.obs.acquisition_ID==ID]#select one acquisition
        celltypes = sub_adata.obs.Pixie.value_counts()[sub_adata.obs.Pixie.value_counts()>50].index.values#take cell types that have at least 50 cells in the acquisition
        sq.gr.spatial_neighbors(sub_adata,coord_type='grid',n_neighs=6,radius = (0,cell_radius))
        sq.gr.nhood_enrichment(sub_adata, cluster_key='Pixie')
        sq.gr.interaction_matrix(sub_adata, cluster_key='Pixie')
        enrichment = sub_adata.uns['Pixie_nhood_enrichment']['zscore']
        a = np.array(sub_adata.obs['Pixie'].cat.categories)#label list
        enrichment = pd.DataFrame(enrichment,index=a,columns=a)
        enrichment = enrichment.loc[celltypes,celltypes]#filter out the cells types that have few cells in the fov
        contact = sub_adata.uns['Pixie_interactions']/sub_adata.uns['Pixie_interactions'].sum()#number of links between cell types
        contact = pd.DataFrame(contact,index=a,columns=a)
        contact = contact.loc[celltypes,celltypes]#filter out the cells types that have few cells in the fov
        store[ID]={'interaction':contact,'enrichment':enrichment}
    return store
cell_table_path=path0+'../segmentation/cell_table_Denoised/cell_table_size_normalized_cell_labels.csv'
biosamples_path=path0+'../IMC_data/ExtraDocs/processed_response.csv'
tb = pd.read_csv(cell_table_path)
cell_radius = tb['major_axis_length'].quantile(0.9)#take the 90% of the cell lenght as a cutoff for cell-cell contact distance, this is around 20 micrometer
adata = generate_anndata_from_ark_analysis(cell_table_path=cell_table_path,biosamples_path=biosamples_path)
core = adata[adata.obs['SAMPLE_TYPE_(CORE/RESECTION)']=='CORE'].copy()

store = neighbours(adata.obs.acquisition_ID.drop_duplicates())#store is a nested dictionary.
with open('neighbours_matrix.pkl', 'wb') as f:
    pickle.dump(store, f)
