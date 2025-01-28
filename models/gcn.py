from sklearn.linear_model import LogisticRegression
import torch.nn as nn
import torch
from torch_geometric.nn import GCNConv, SSGConv, global_sort_pool, TopKPooling, global_mean_pool, GATConv

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import torch.nn.functional as F
import numpy as np
import pdb
import wandb
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt
from mainutils.utils import visualise_cellgraph
import io
from PIL import Image
from mainutils.utils import coords_to_graph


import torch
import torch.nn as nn
import torch.nn.functional as F
from models.graph_networks import GCN, SSGCN, SAGEAttentionNet, GCNWithAttention

GCN_DICT = {
			'gcn': GCN,
			'ssgcn': SSGCN,
			'sagegcn': SAGEAttentionNet,
			'gcnattn': GCNWithAttention,
}

class GraphConvolutionalNetwork:
	"""
	Class for training GCN models on TNBC (Triple-Negative Breast Cancer) expression and spatial data.

	Args:
		logger (optional, Logger): Logger object for logging training information.
	"""

	def __init__(self,
				gconv, 
				lr,
				nm_epochs,
				batch_size,
				fnorm,
				logger=None,
				device='cpu',
				class_weight=None,
				gmethod='knn',
				radius=7,
				batch_correct=False,
				nm_batch=6,
				**kwargs):
		"""
		Initializes the TNBC GCN model and optimizer.

		Args:
			logger (optional, Logger): Logger object for logging training information.
		"""
		self.gconv = gconv
		self.nm_epochs = nm_epochs
		self.batch_size = batch_size
		self.fnorm = fnorm
		self.logger = logger
		self.class_weight = class_weight
		self.gmethod = gmethod
		self.radius = radius
		self.batch_correct = batch_correct
		self.nm_batch = nm_batch

		self.device = torch.device(device)
		gcn_params = kwargs.get(self.gconv, None)	
		self.model = GCN_DICT[self.gconv](**gcn_params).to(self.device)
		
		self.optim = torch.optim.AdamW(
					self.model.parameters(), 
					lr=lr, 
					weight_decay=1e-4,
					betas=(0.9, 0.999)
			)


		weights = None if class_weight is None else torch.tensor(
				[weight for target, weight in class_weight[1]],
				dtype=torch.float32
				).to(self.device)

		#self.criterion = nn.CrossEntropyLoss(weight=weights)
		self.criterion = nn.BCEWithLogitsLos(pos_weight=weights)

	def to_pyg(self, data_dict):
		"""
			Converts input data (gene expression, graphs, and optional labels) into PyTorch Geometric Data objects.

		Args:
			data (dict): A dictionary containing: expressions, enrichments, graphs, labels, and markers.
			expressions (list): List of gene expression data for each sample.
			coords (list): List of coords of cells.
			labels (list): List of labels (0 or 1) for each sample.
			markers (list): List of feature names. 

		Returns:
			dataset (list): List of PyTorch Geometric Data objects representing the samples.
		"""
		dataset = []
		num_samples = len(data_dict['labels'])
		for i in range(num_samples):
			graph_attributes = torch.tensor(data_dict['expressions'][i]).float()
			if self.fnorm == 'minmax':
				graph_attributes = torch.log1p(graph_attributes)
				min_marker, _ = torch.min(graph_attributes, dim=0)
				max_marker, _ = torch.max(graph_attributes, dim=0)
				graph_attributes = (graph_attributes - min_marker)/(max_marker - min_marker + 1e-8)
			elif self.fnorm == 'log1p':
				graph_attributes = torch.log1p(graph_attributes)
				graph_attributes = (graph_attributes - torch.mean(graph_attributes, dim=0, keepdim=True))/(torch.std(graph_attributes, dim=0, keepdim=True) + 1e-8)
			elif self.fnorm == 'arctan':
				graph_attributes = torch.arctan(graph_attributes)
				graph_attributes = (graph_attributes - torch.mean(graph_attributes, dim=0, keepdim=True))/(torch.std(graph_attributes, dim=0, keepdim=True) + 1e-8)
			elif self.fnorm == 'znorm':
				graph_attributes = (graph_attributes - torch.mean(graph_attributes, dim=0, keepdim=True))/(torch.std(graph_attributes, dim=0, keepdim=True) + 1e-8)
			elif self.fnorm == 'raw':
				pass
			else:
				assert 0, f"{self.fnorm} not implemented"
			coords = data_dict['coords'][i]
			graph = coords_to_graph(coords, gmethod=self.gmethod, radius=self.radius)
			graph = graph.tocoo()
			label = torch.tensor(data_dict['labels'][i]).float()
			row, col, data = graph.row, graph.col, graph.data
			edge_index = torch.tensor((row, col)).long()
			edge_attr = torch.tensor(data).float()

			data_pyg = Data(x=graph_attributes, edge_index=edge_index, edge_attr=edge_attr, y=label)

			dataset.append(data_pyg)
		return dataset

	def fit(self, data):
		"""
		Trains the GCN model on the provided data.

		Args:
			data (dict): A dictionary containing: expressions, enrichments, graphs, labels, and markers.
			expressions (list): List of gene expression data for each sample.
			graphs (list): List of spatial adjacency matrices representing connections between samples.
			labels (list): List of labels (0 or 1) for each sample.
		"""
		self.model.train()
		dataset = self.to_pyg(data)
		loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
		for epoch in range(self.nm_epochs):
			loss_epoch = 0.
			for x_batch in loader:
				x_batch = x_batch.to(self.device)
				self.optim.zero_grad()
				logits, latent_z = self.model.hidden_representation(
					x_batch.x, 
					x_batch.edge_index, 
					x_batch.edge_attr, 
					x_batch.batch
				)
				
				loss = self.criterion(logits, x_batch.y.to(torch.int64))

				loss.backward()
				torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
				self.optim.step()

				loss_epoch += loss.item()
				self.logger.log({'GCN Iteration Loss': loss.item()})
				
			loss_epoch = loss_epoch/(len(loader))
			self.logger.log({'GCN Epoch Loss': loss.item()})

	def predict(self, data, threshold=0.5):
		"""
		Predicts class labels using the trained GCN model.

		Args:
			X (list): List of gene expression data for each sample.
			graphs (list): List of spatial adjacency matrices representing connections between samples.

		Returns:
			preds (np.array) : Array of predicted class labels (0 or 1) for each sample.
		"""
		self.model.eval()

		with torch.no_grad():
			dataset = self.to_pyg(data)
			loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
			preds = np.array([])
			for x_batch in loader:
				x_batch = x_batch.to(self.device)
				preds_i, latent_z = self.model.hidden_representation(x_batch.x, x_batch.edge_index, x_batch.edge_weight, x_batch.batch)
				#preds_i = F.softmax(preds_i, dim=1)
				#preds_i = preds_i.argmax(dim=1).cpu().numpy()
				preds_i = F.sigmoid(preds_i).cpu().numpy()
				preds = np.concatenate([preds, preds_i])
			preds = (preds>=threshold).astype(int)
			return preds

	def predict_proba(self, data):
		"""
		Predicts class probabilities using the trained GCN model.

		Args:
			data (dict): A dictionary containing: expressions, enrichments, graphs, labels, and markers.
			expressions (list): List of gene expression data for each sample.
			graphs (list): List of spatial adjacency matrices representing connections between samples.
			labels (list): List of labels (0 or 1) for each sample.
			markers (list): List of feature names. 
		Returns:
			preds (np.array) : Array of predicted probability for each sample.
		"""
		self.model.eval()

		with torch.no_grad():
			pyg_dataset = self.to_pyg(data)
			loader = DataLoader(pyg_dataset, batch_size=self.batch_size, shuffle=False)
			probs = []
			for x_batch in loader:
				x_batch = x_batch.to(self.device)
				score, latent_z = self.model.hidden_representation(x_batch.x, x_batch.edge_index, x_batch.edge_attr, x_batch.batch)
				#score = F.softmax(score, dim=1)
				probs = F.sigmoid(score)
				probs.append(score)
			probs = torch.cat(probs, dim=0).cpu().numpy()
			return probs

	def wandb_log_figure(self, fig, name):
		"""
		Helper to log a matplotlib figure to W&B or any logger with .log(...).
		"""
		buf = io.BytesIO()
		fig.savefig(buf, format='png')
		buf.seek(0)
		if self.logger:
			self.logger.log({name: wandb.Image(Image.open(buf))})
		plt.close(fig)

	def visualize_latent_space(self, data, method='PCA'):
		from sklearn.decomposition import PCA
		from sklearn.preprocessing import StandardScaler
		import umap

		with torch.no_grad():
			dataset = self.to_pyg(data)
			loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
			all_rois, all_labels = [], []
			for x_batch in loader:
				x_batch = x_batch.to(self.device)
				_, latent_z = self.model.hidden_representation(x_batch.x, x_batch.edge_index, x_batch.edge_weight, x_batch.batch)
				all_rois.append(latent_z)
				all_labels.append(x_batch.stain_y)
			all_rois = torch.cat(all_rois, dim=0).cpu().numpy()
			all_labels = torch.cat(all_labels, dim=0).cpu().numpy()

		if method == 'PCA':
			scaler = StandardScaler()
			latent_patients = scaler.fit_transform(all_rois)
			reducer = PCA(n_components=2)
			projections = reducer.fit_transform(all_rois)
		elif method == 'UMAP':
			reducer = umap.UMAP()
			projections = reducer.fit_transform(all_rois)
		else:
			assert 0,'Not Implemented'
		
		unique_labels = np.unique(all_labels)
		colors = plt.cm.get_cmap('tab10', len(unique_labels))
		fig, ax = plt.subplots()

		for i, label in enumerate(unique_labels):
			mask = all_labels == label
			ax.scatter(projections[mask, 0], projections[mask, 1], color=colors(i), label=label)
		ax.set_title(f"ROI Clustering Visualization ({method})")
		ax.set_xlabel(f"{method} 1")
		ax.set_ylabel(f"{method} 2")
		ax.legend()
		self.wandb_log_figure(fig, f"Latent Encoding {method}")
		
	def pyg_attribution(self, data, topk=10):
		from torch_geometric.explain import Explainer, GNNExplainer
		import torch_geometric.utils as utils
		self.model.eval()
		explainer = Explainer(
						model=self.model,
						algorithm=GNNExplainer(epochs=100),
						explanation_type='phenomenon',
						node_mask_type='attributes',
						edge_mask_type='object',
						model_config=dict(
						mode='multiclass_classification',
						task_level='graph',
						return_type='log_probs',
						),
					)

		pyg_dataset = self.to_pyg(data)
		loader = DataLoader(pyg_dataset, batch_size=1, shuffle=False)
		
		feature_names = np.array(data['markers'])
		celltypes = data.get('celltypes', [])

		top_k_count_celltypespositive = {name: 0 for name in celltypes}
		top_k_count_celltypesnegative = {name: 0 for name in celltypes}

		top_k_count_positive = {name: 0 for name in feature_names}
		top_k_count_negative = {name: 0 for name in feature_names}

		labels_dict = {1:'Responder', 0:'Non-Responder'}
		
		for i, x_batch in enumerate(loader):
			x_batch = x_batch.to(self.device)

			kwargs = {'edge_weight':x_batch.edge_attr, 'batch': x_batch.batch}
			if hasattr(x_batch, 'batch'):
				kwargs['batch'] = x_batch.batch

			explanation = explainer(
					x_batch.x, 
					x_batch.edge_index, 
					target=x_batch.y.long(), 
					**kwargs
				)

			nm = explanation.node_mask
			em = explanation.edge_mask

			node_thresh = torch.quantile(nm, 0.85)
			edge_thresh = torch.quantile(em, 0.85)

			node_mask = (nm > node_thresh)
			edge_mask = (em > edge_thresh)

			# Feature importance
			scores = explanation.node_mask.mean(0).cpu().numpy()
			sorted_indices = scores.argsort()[::-1][:topk]
			sorted_scores = scores[sorted_indices]
			sorted_feature_names = feature_names[sorted_indices]

			label_val = int(x_batch.y.item())
			for feature in sorted_feature_names:
				if label_val == 1:
					top_k_count_positive[feature] += 1
				else:
					top_k_count_negative[feature] += 1

			sub_graph = x_batch.edge_index[:,edge_mask]
			retained_nodes = torch.nonzero((node_mask.sum(1) != 0), as_tuple=False).squeeze()

			src_dst = sub_graph.t().tolist()
			mask = [(src in retained_nodes) and (dst in retained_nodes) for src, dst in src_dst]
			mask = torch.tensor(mask)
			filtered_sub_graph = sub_graph[:, mask]
			unique_nodes = torch.unique(filtered_sub_graph)

			
			fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 12))
			
			coo_adj_full = utils.to_scipy_sparse_matrix(
				x_batch.edge_index, 
				edge_attr=x_batch.edge_attr, 
				num_nodes=x_batch.x.shape[0]
			)
			if hasattr(x_batch, 'cell_labels'):
				node_labels = x_batch.cell_labels
			else:
				node_labels = None

			csr_adj_full = coo_adj_full.tocsr()

			_, _, pos, label_to_color, _ = visualise_cellgraph(csr_adj_full, random_state=42, node_labels=x_batch.cell_labels, show=False, spatial_coords=data['coords'][i], ax=ax1, add_legend=True)

			coo_adj_sub = utils.to_scipy_sparse_matrix(
				filtered_sub_graph, 
				edge_attr= torch.ones(filtered_sub_graph.size(1), dtype=torch.float), 
				num_nodes=x_batch.x.shape[0])
			csr_adj_sub = coo_adj_sub.tocsr()

			_, _, _, _, freq = visualise_cellgraph(csr_adj_sub, random_state=42, node_labels=x_batch.cell_labels, show=False, ax=ax2, pos=pos, label_to_color=label_to_color, largest_comp=True)

			self.wandb_log_figure(fig, f"results/SubGraph_{data['leapid'][i]}_{data['patient'][i]}_{labels_dict[x_batch.y.cpu().int().item()]}")


			fig2 = plt.figure(figsize=(32, 10))
			ax_left = fig2.add_subplot(1, 2, 1)
			ax_right = fig2.add_subplot(1, 2, 2)

			# Full Graph
			ax_left.bar(range(len(sorted_scores)), sorted_scores, tick_label=sorted_feature_names)
			ax_left.set_xlabel('Proteins', fontsize=28)
			ax_left.set_ylabel('Importance Score', fontsize=28)
			ax_left.set_title(f"GCN {labels_dict[x_batch.y.cpu().int().item()]} {data['patient'][i]}", fontsize=32)
			ax_left.tick_params(axis='x', labelrotation=45, labelsize=28)

			avg_expr = data['expressions'][i].mean(0)
			topk_expr = avg_expr[sorted_indices]
			# Subgraph
			ax_right.bar(range(len(topk_expr)), topk_expr, tick_label=sorted_feature_names)
			ax_right.set_xlabel('Proteins', fontsize=28)
			ax_right.set_ylabel('Average Expression Across Cells', fontsize=28)
			ax_right.set_title(f"GCN {labels_dict[x_batch.y.cpu().int().item()]} {data['patient'][i]}", fontsize=32)
			ax_right.tick_params(axis='x', labelrotation=45, labelsize=28)

			plt.tight_layout()
			self.wandb_log_figure(fig2, f"GNNExplainer ROI {data['leapid'][i]} {data['patient'][i]}")

			if label_val == 1:
				for celltype, count in freq.items():
					top_k_count_celltypespositive[celltype] += count
			else:
				for celltype, count in freq.items():
					top_k_count_celltypesnegative[celltype] += count			

		fig3, ax3 = plt.subplots(figsize=(12, 6))
		indices = np.arange(len(top_k_count_positive))
		# Plot the bars
		ax3.bar(indices, top_k_count_positive.values(), 0.35, label='Responder', color='skyblue')
		ax3.bar(indices + 0.35, top_k_count_negative.values(), 0.35, label='Non-responder', color='salmon')

		# Add some text for labels, title and axes ticks
		ax3.set_xlabel('Features', fontsize=14)
		ax3.set_ylabel(f"Top  {topk} Frequency", fontsize=14)
		ax3.set_title(f"Top {topk} Markers for Responder vs Non-Responder", fontsize=16)
		ax3.set_xticks(indices + 0.35 / 2)
		ax3.set_xticklabels(list(top_k_count_positive.keys()), rotation=45, ha='right')
		ax3.legend()
		plt.tight_layout()
		
		self.wandb_log_figure(fig3, 'Most Frequent Marker GNNExplainer across ROIs')


		fig4, ax4 = plt.subplots(figsize=(12, 6))
		indices = np.arange(len(top_k_count_celltypespositive))
		# Plot the bars
		val_pos = [el/(i+1) for el in list(top_k_count_celltypespositive.values())]
		val_neg = [el/(i+1) for el in list(top_k_count_celltypespositive.values())]
		
		ax4.bar(indices, val_pos, 0.35, label='Responder', color='skyblue')
		ax4.bar(indices + 0.35, val_neg, 0.35, label='Non-responder', color='salmon')

		# Add some text for labels, title and axes ticks
		ax4.set_xlabel('Features', fontsize=14)
		ax4.set_ylabel(f"Top  {topk} Frequency", fontsize=14)
		ax4.set_title(f"Top {topk} Markers for Responder vs Non-Responder", fontsize=16)
		ax4.set_xticks(indices + 0.35 / 2)
		ax4.set_xticklabels(list(top_k_count_celltypespositive.keys()), rotation=45, ha='right')
		ax4.legend()
		plt.tight_layout()
		self.wandb_log_figure(fig3, 'Most Frequent Celltypes GNNExplainer across ROIs')


	def gradient_attribution(self, data, topk=10, target_class=1):
		feature_names = data['markers']
		import matplotlib.pyplot as plt
		self.model.eval()
		pyg_dataset = self.to_pyg(data)
		loader = DataLoader(pyg_dataset, batch_size=1, shuffle=False)
		avg_node_gradients, std_node_gradients = [], []
		y_all = []
		for i, x_batch in enumerate(loader):
			x_batch = x_batch.to(self.device)
			self.model.zero_grad()
			x_batch.x.requires_grad = True
			logits = self.model.hidden_representation(x_batch.x, x_batch.edge_index, x_batch.edge_attr, x_batch.batch)
			logits[target_class].backward()
			y_all.append(x_batch.y)
			node_gradients = x_batch.x.grad

			avg_node_gradients.append(node_gradients.abs().mean(0))
			std_node_gradients.append(node_gradients.abs().std(0))

			node_gradients = node_gradients.T.abs().cpu().numpy() 
			top_k_idx = node_gradients.argsort(1)[:,-topk:]
			top_k_value = node_gradients[np.arange(node_gradients.shape[0])[:, None], top_k_idx]
			plt.imshow(top_k_value, cmap='hot', aspect='auto')
			plt.xlabel('Node Index')
			plt.ylabel('Attribute Index')
			plt.yticks(ticks=range(len(feature_names)), labels=feature_names, fontsize=8)
			#plt.xticks(range(len(feature_names)), feature_names, rotation=90)
			plt.title('Gradient of Node Attributes with Respect to log prob of a responder')
			plt.tight_layout()
			plt.colorbar(label='Gradient')
			self.logger.log({f"Attribution GCN batch {i}": plt})
			plt.clf()
			plt.cla()
		avg_node_gradients = torch.stack(avg_node_gradients).cpu().numpy()
		std_node_gradients = torch.stack(std_node_gradients).cpu().numpy()
		y_all = torch.stack(y_all).cpu().squeeze().numpy().astype('int')

		avg_node_gradients_resp = avg_node_gradients[y_all].mean(0)
		std_node_gradients_resp = std_node_gradients[y_all].mean(0)
		avg_node_gradients_noresp = avg_node_gradients[1 - y_all].mean(0)
		std_node_gradients_noresp = std_node_gradients[1 - y_all].mean(0)

		grad_df = pd.DataFrame({'Feature': np.array(feature_names), 
									'Mean Gradients pCR': avg_node_gradients_resp,
									'Std Gradients pCR': std_node_gradients_resp,
									'Mean Gradients Non-Responder': avg_node_gradients_noresp,
									'Std Gradients Non-Responder': std_node_gradients_noresp,
								})
		grad_df.set_index('Feature', inplace=True)
		fig = plt.figure(figsize=(16, 10))
		x = np.arange(len(feature_names))
		bar_width = 0.35
		plt.bar(x - bar_width/2, grad_df['Mean Gradients pCR'], yerr=grad_df['Std Gradients pCR'], width=bar_width, label='Responder', capsize=5, alpha=0.7)
		plt.bar(x + bar_width/2, grad_df['Mean Gradients pCR'], yerr=grad_df['Std Gradients Non-Responder'], width=bar_width, label='Non-Responder', capsize=5, alpha=0.7)
		#grad_df.plot(kind='bar', rot=0)
		#table = wandb.Table(dataframe=grad_df)
		# plt.xlabel('Protein Expres', fontsize=12)
		plt.ylabel('Gradient Averaged Across ROIs', fontsize=28)
		plt.title('Comparison of Gradients Averaged Across ROIs', fontsize=32)
		plt.legend(fontsize=12, prop={'size': 8})
		plt.xticks(ticks=np.arange(len(feature_names)), labels=grad_df.index, rotation=45, ha='right')
		plt.tight_layout()
		buffer = io.BytesIO()
		buffer.seek(0)
		plt.savefig(buffer, format='png')
		self.logger.log({'Average gradients across ROIs': wandb.Image(Image.open(buffer))})

