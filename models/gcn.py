from sklearn.linear_model import LogisticRegression
import torch.nn as nn
import torch
from torch_geometric.nn import GCNConv, global_mean_pool, SSGConv
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
from models.adversarial_batch import AdversarialClassifier

class GCN(nn.Module):
	"""
	Graph Convolutional Network (GCN) model for node classification.

	Args:
		input_dim (int): Dimensionality of the input node features.
		hidden_dim (int): Dimensionality of the hidden layer.
		nm_class (int): Number of classes for node classification.
	"""
	def __init__(self, 
					input_dim, 
					hidden_dim, 
					nm_class):
		super(GCN, self).__init__()
		self.input_dim = input_dim
		self.hidden_dim = hidden_dim
		self.nm_class = nm_class
		self.conv1 = GCNConv(self.input_dim, self.hidden_dim)
		self.conv2 = GCNConv(self.hidden_dim, self.hidden_dim)
		self.clf = nn.Linear(self.hidden_dim, self.nm_class)

	def hidden_representation(self, x, edge_index, edge_weight, batch):
		x = self.conv1(x, edge_index, edge_weight)
		x = F.relu(x)
		x = self.conv2(x, edge_index, edge_weight)
		x = global_mean_pool(x, batch)
		return self.clf(x), x

	def forward(self, x, edge_index, edge_weight, batch):
		x = self.conv1(x, edge_index, edge_weight)
		x = F.relu(x)
		x = self.conv2(x, edge_index, edge_weight)
		x = global_mean_pool(x, batch)
		return self.clf(x)

class SSGCN(nn.Module):
	"""
	Simple Spectral Graph Convolution (SSGCN) model for node classification.

	Args:
		input_dim (int): Dimensionality of the input node features.
		hidden_dim (int): Dimensionality of the hidden layer.
		nm_class (int): Number of classes for node classification.
	"""
	def __init__(self, 
					input_dim, 
					hidden_dim,
					K,
					alpha, 
					nm_class=1):
		super(SSGCN, self).__init__()
		self.input_dim = input_dim
		self.hidden_dim = hidden_dim
		self.nm_class = nm_class
		self.conv1 = SSGConv(self.input_dim, self.hidden_dim, self.K, self.alpha)
		self.conv2 = SSGConv(self.hidden_dim, self.hidden_dim, self.K, self.alpha)
		self.clf = nn.Linear(self.hidden_dim, self.nm_class)

	def hidden_representation(self, x, edge_index, edge_weight, batch):
		x = self.conv1(x, edge_index, edge_weight)
		x = F.relu(x)
		x = self.conv2(x, edge_index, edge_weight)
		x = global_mean_pool(x, batch)
		return self.clf(x), x

	def forward(self, x, edge_index, edge_weight, batch):
		x = self.conv1(x, edge_index, edge_weight)
		x = F.relu(x)
		x = self.conv2(x, edge_index, edge_weight)
		x = global_mean_pool(x, batch)
		return self.clf(x)

GCN_DICT = {
			'gcn': GCN,
			'ssgcn': SSGCN,
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
		self.optim = torch.optim.Adam(self.model.parameters(), lr=lr)
		if class_weight is None:
			weights = None
		else:
			weights = [weight for target, weight in class_weight.items()]
			weights = torch.tensor(weights, dtype=torch.float32).to(self.device)
		self.criterion = nn.CrossEntropyLoss(weight=weights)
		if self.batch_correct:
			#self.batch_tf = nn.Sequential(nn.Linear(gcn_params['input_dim'], gcn_params['input_dim']),
			#							nn.ReLU()).to(self.device)
			self.batch_pred = AdversarialClassifier(gcn_params['hidden_dim'], self.nm_batch).to(self.device)
			self.optim_b = torch.optim.Adam(self.batch_pred.parameters(), lr=lr)
			#self.optim_b = torch.optim.Adam(list(self.batch_pred.parameters())+list(self.batch_tf.parameters()), lr=lr)

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
			#if self.batch_correct:
			stain_y = torch.tensor(data_dict['stain'][i]).float()
			data_pyg = Data(x=graph_attributes, edge_index=edge_index, edge_attr=edge_attr, y=label, stain_y=stain_y)
			#else:
			#	data_pyg = Data(x=graph_attributes, edge_index=edge_index, edge_attr=edge_attr, y=label)
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
		if self.batch_correct: self.batch_pred.train()

		dataset = self.to_pyg(data)
		loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
		for epoch in range(self.nm_epochs):
			loss_epoch = 0.
			for x_batch in loader:
				x_batch = x_batch.to(self.device)
				self.optim.zero_grad()
				logits, latent_z = self.model.hidden_representation(x_batch.x, x_batch.edge_index, x_batch.edge_attr, x_batch.batch)
				loss = self.criterion(logits, x_batch.y)
				if self.batch_correct and epoch<50:
					self.optim_b.zero_grad()
					pred_batch = self.batch_pred(latent_z)
					loss_adv_batch = self.batch_pred.loss(pred_batch, x_batch.stain_y)
					loss_f = loss + loss_adv_batch
					loss_f.backward()
					self.optim_b.step()
					self.logger.log({'GCN Adversarial Batch Loss': loss_adv_batch.item()})
					self.logger.log({'GCN Full Loss': loss_f.item()})
					loss_epoch += loss_f.item()
				else:
					loss.backward()
					loss_epoch += loss.item()

				self.optim.step()
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
		if self.batch_correct: self.batch_pred.eval()

		with torch.no_grad():
			dataset = self.to_pyg(data)
			loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
			preds = np.array([])
			b_preds = np.array([]) if self.batch_correct else None
			for x_batch in loader:
				x_batch = x_batch.to(self.device)
				preds_i, latent_z = self.model.hidden_representation(x_batch.x, x_batch.edge_index, x_batch.edge_weight, x_batch.batch)
				preds_i = F.softmax(preds_i, dim=1)
				preds_i = preds_i.argmax(dim=1).cpu().numpy()
				preds = np.concatenate([preds, preds_i])
				if self.batch_correct:
					b_preds_i = self.batch_pred(latent_z)
					b_preds_i = F.softmax(b_preds_i, dim=1)
					b_preds_i = b_preds_i.argmax(dim=1).cpu().numpy()
					b_preds = np.concatenate([b_preds, b_preds_i])
			return preds, b_preds

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
		if self.batch_correct: self.batch_pred.eval()

		with torch.no_grad():
			pyg_dataset = self.to_pyg(data)
			loader = DataLoader(pyg_dataset, batch_size=self.batch_size, shuffle=False)
			preds = []
			b_preds = [] if self.batch_correct else None
			for x_batch in loader:
				x_batch = x_batch.to(self.device)
				score, latent_z = self.model.hidden_representation(x_batch.x, x_batch.edge_index, x_batch.edge_attr, x_batch.batch)
				score = F.softmax(score, dim=1)
				preds.append(score)
				if self.batch_correct:
					b_score = self.batch_pred(latent_z)
					b_score = F.softmax(b_score, dim=1)
					b_preds.append(b_score)
			preds = torch.cat(preds, dim=0).cpu().numpy()
			if self.batch_correct:
				b_preds = torch.cat(b_preds, dim=0).cpu().numpy()
			return preds, b_preds

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
		#fig.savefig('Latent_Encoding_withoutadversary_umap.png')
		buffer = io.BytesIO()
		buffer.seek(0)
		fig.savefig(buffer, format='png')
		self.logger.log({f"Latent Encoding {method}": wandb.Image(Image.open(buffer))})
		
	def pyg_attribution(self, data, topk=10):
		from torch_geometric.explain import Explainer, GNNExplainer
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
		top_k_count_positive, top_k_count_negative = {name: 0 for name in feature_names}, {name: 0 for name in feature_names}
		labels_dict = {1:'Responder', 0:'Non-Responder'}
		for i, x_batch in enumerate(loader):
			x_batch.cell_labels = data['cell_labels'][i]
			x_batch = x_batch.to(self.device)
			kwargs = {'edge_weight':x_batch.edge_attr, 'batch': x_batch.batch}
			explanation = explainer(x_batch.x, x_batch.edge_index, target=x_batch.y.long(), **kwargs)

			node_mask = (explanation.node_mask>torch.quantile(explanation.node_mask, 0.85))
			edge_mask = (explanation.edge_mask>torch.quantile(explanation.edge_mask, 0.85))

			sub_graph = x_batch.edge_index[:,edge_mask]

			scores = explanation.node_mask.mean(0).cpu().numpy()
			sorted_indices = scores.argsort()[::-1]
			sorted_indices = sorted_indices[:topk]
			sorted_scores = scores[sorted_indices]
			sorted_feature_names = feature_names[sorted_indices]

			for feature in sorted_feature_names:
				if int(x_batch.y.cpu().item()) == 1:
					top_k_count_positive[feature] += 1
				else:
					top_k_count_negative[feature] += 1

			sorted_expression = data['expressions'][i].mean(0)[sorted_indices]
			nodes_retained = torch.nonzero((node_mask.sum(1)!=0).cpu(), as_tuple=False).squeeze()

			mask = [(src in nodes_retained) and (dst in nodes_retained) for src, dst in sub_graph.t().tolist()]
			mask = torch.tensor(mask)
			filtered_sub_graph = sub_graph[:, mask]
			unique_nodes = torch.unique(filtered_sub_graph)
			import torch_geometric.utils as utils
			if i<40:
				fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 12))
				coo_adj_full = utils.to_scipy_sparse_matrix(x_batch.edge_index, edge_attr=x_batch.edge_attr, num_nodes=x_batch.x.shape[0])
				csr_adj_full = coo_adj_full.tocsr()

				_, _, pos, label_to_color = visualise_cellgraph(csr_adj_full, random_state=42, node_labels=x_batch.cell_labels, show=False, spatial_coords=data['coords'][i], ax=ax1, add_legend=True)

				coo_adj_sub = utils.to_scipy_sparse_matrix(filtered_sub_graph, edge_attr= torch.ones(filtered_sub_graph.size(1), dtype=torch.float), num_nodes=x_batch.x.shape[0])
				csr_adj_sub = coo_adj_sub.tocsr()

				_, _, _, _ = visualise_cellgraph(csr_adj_sub, random_state=42, node_labels=x_batch.cell_labels, show=False, ax=ax2, pos=pos, label_to_color=label_to_color, largest_comp=True)

				buffer = io.BytesIO()
				buffer.seek(0)
				plt.savefig(buffer, format='png')
				self.logger.log({f"results/SubGraph_{i}_{data['patient'][i]}_{labels_dict[x_batch.y.cpu().int().item()]}": wandb.Image(Image.open(buffer))})

				plt.cla()
				plt.clf()
				plt.close()

				plt.figure(figsize=(32, 10))
				# Full Graph
				plt.subplot(1, 2, 1)
				plt.bar(range(len(sorted_scores)), sorted_scores, tick_label=sorted_feature_names)
				plt.xlabel('Proteins', fontsize=28)
				plt.ylabel('Importance Score', fontsize=28)
				plt.title(f"GCN {labels_dict[x_batch.y.cpu().int().item()]} {data['patient'][i]}", fontsize=32)
				plt.xticks(rotation=45, ha='right', fontsize=28)
				plt.subplots_adjust(bottom=0.2)
				plt.tight_layout()

				# Subgraph
				plt.subplot(1, 2, 2)
				plt.bar(range(len(sorted_scores)), sorted_expression, tick_label=sorted_feature_names)
				plt.xlabel('Proteins', fontsize=28)
				plt.ylabel('Average Expression Across Cells', fontsize=28)
				plt.title(f"GCN {labels_dict[x_batch.y.cpu().int().item()]} {data['patient'][i]}", fontsize=32)
				plt.xticks(rotation=45, ha='right', fontsize=28)
				plt.subplots_adjust(bottom=0.2)
				plt.tight_layout()


				buffer = io.BytesIO()
				buffer.seek(0)
				plt.savefig(buffer, format='png')
				self.logger.log({f"GNNExplainer ROI {i} {data['patient'][i]}": wandb.Image(Image.open(buffer))})
				plt.close()

		fig, ax = plt.subplots(figsize=(12, 6))
		indices = np.arange(len(top_k_count_positive))
		# Plot the bars
		bars_positive = ax.bar(indices, top_k_count_positive.values(), 0.35, label='Responder', color='skyblue')
		bars_negative = ax.bar(indices + 0.35, top_k_count_negative.values(), 0.35, label='Non-responder', color='salmon')

		# Add some text for labels, title and axes ticks
		ax.set_xlabel('Features', fontsize=14)
		ax.set_ylabel(f"Top  {topk} Frequency", fontsize=14)
		ax.set_title(f"Top {topk} Markers for Responder vs Non-Responder", fontsize=16)
		ax.set_xticks(indices + 0.35 / 2)
		ax.set_xticklabels(list(top_k_count_positive.keys()), rotation=45, ha='right')
		ax.legend()
		plt.tight_layout()

		buffer = io.BytesIO()
		buffer.seek(0)
		plt.savefig(buffer, format='png')
		self.logger.log({'Most Frequent Marker GNNExplainer across ROIs': wandb.Image(Image.open(buffer))})
		plt.close()

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

