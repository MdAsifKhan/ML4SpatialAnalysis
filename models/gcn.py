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
		self.conv1 = GCNConv(input_dim, hidden_dim)
		self.conv2 = GCNConv(hidden_dim, hidden_dim)
		self.clf = nn.Linear(hidden_dim, nm_class)

	def forward(self, x, edge_index, edge_weight, batch):
		x = self.conv1(x, edge_index, edge_weight)
		x = F.relu(x)
		x = self.conv2(x, edge_index, edge_weight)
		x = global_mean_pool(x, batch)
		return F.sigmoid(self.clf(x))


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
		self.conv1 = SSGConv(input_dim, hidden_dim, K, alpha)
		self.conv2 = SSGConv(hidden_dim, hidden_dim, K, alpha)
		self.clf = nn.Linear(hidden_dim, nm_class)

	def forward(self, x, edge_index, edge_weight, batch):
		x = self.conv1(x, edge_index, edge_weight)
		x = F.relu(x)
		x = self.conv2(x, edge_index, edge_weight)
		x = global_mean_pool(x, batch)
		return F.sigmoid(self.clf(x))

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

		self.device = torch.device(device)
		self.model = GCN_DICT[self.gconv](**kwargs.get(self.gconv, None)).to(self.device)
		self.optim = torch.optim.Adam(self.model.parameters(), lr=lr)
		self.criterion = nn.BCELoss()

	def to_pyg(self, data_dict):
		"""
			Converts input data (gene expression, graphs, and optional labels) into PyTorch Geometric Data objects.

		Args:
			data (dict): A dictionary containing: expressions, enrichments, graphs, labels, and markers.
			expressions (list): List of gene expression data for each sample.
			graphs (list): List of spatial adjacency matrices representing connections between samples.
			labels (list): List of labels (0 or 1) for each sample.
			markers (list): List of feature names. 

		Returns:
			dataset (list): List of PyTorch Geometric Data objects representing the samples.
		"""
		dataset = []
		num_samples = len(data_dict['labels'])
		for i in range(num_samples):
			# import wandb
			# wandb.finish()
			# import pdb
			# pdb.set_trace()
			graph_attributes = torch.tensor(data_dict['expressions'][i]).float()
			if self.fnorm == 'minmax':
				graph_attributes = torch.log1p(graph_attributes)
				min_marker, _ = torch.min(graph_attributes, dim=0)
				max_marker, _ = torch.max(graph_attributes, dim=0)
				graph_attributes = (graph_attributes - min_marker)/(max_marker - min_marker + 1e-8)
			elif self.fnorm == 'log1p':
				graph_attributes = torch.log1p(graph_attributes)
				graph_attributes = (graph_attributes - torch.mean(graph_attributes, dim=0, keepdim=True))/(torch.std(graph_attributes, dim=0, keepdim=True) + 1e-8)
			elif self.fnorm == 'znorm':
				graph_attributes = (graph_attributes - torch.mean(graph_attributes, dim=0, keepdim=True))/(torch.std(graph_attributes, dim=0, keepdim=True) + 1e-8)
			elif self.fnorm == 'raw':
				pass
			else:
				assert 0, f"{self.fnorm} not implemented"
			graph = data_dict['graphs'][i]
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
				logits = self.model(x_batch.x, x_batch.edge_index, x_batch.edge_attr, x_batch.batch)
				y_batch = x_batch.y.unsqueeze(1)
				loss = self.criterion(logits, y_batch)
				loss.backward()
				self.optim.step()
				self.logger.log({'GCN Iteration Loss': loss.item()})
				loss_epoch += loss.item()
			loss_epoch = loss_epoch/(len(loader))
			self.logger.log({'GCN Epoch Loss': loss.item()})

	def predict(self, data):
		"""
		Predicts class labels using the trained GCN model.

		Args:
			X (list): List of gene expression data for each sample.
			graphs (list): List of spatial adjacency matrices representing connections between samples.

		Returns:
			preds (np.array) : Array of predicted class labels (0 or 1) for each sample.
		"""
		self.model.eval()
		dataset = self.to_pyg(data)
		loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
		preds = np.array([])
		for x_batch in loader:
			x_batch = x_batch.to(self.device)
			preds_i = self.model(x_batch.x, x_batch.edge_index, x_batch.edge_weight, x_batch.batch)
			preds_i = (preds_i.squeeze().detach().cpu().numpy()>0.5).astype(float)
			preds = np.concatenate([preds, preds_i])
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
		pyg_dataset = self.to_pyg(data)
		loader = DataLoader(pyg_dataset, batch_size=self.batch_size, shuffle=False)
		preds = np.array([])
		for x_batch in loader:
			x_batch = x_batch.to(self.device)
			# x_batch
			score = self.model(x_batch.x, x_batch.edge_index, x_batch.edge_attr, x_batch.batch)
			preds_i = score.squeeze().detach().cpu().numpy()
			preds = np.concatenate([preds, preds_i])
		return preds

	def pyg_attribution(self, data):
		from torch_geometric.explain import Explainer, GNNExplainer
		self.model.eval()
		explainer = Explainer(
						model=self.model,
						algorithm=GNNExplainer(epochs=100),
						explanation_type='model',
						node_mask_type='attributes',
						edge_mask_type='object',
						model_config=dict(
						mode='binary_classification',
						task_level='graph',
						return_type='probs',
						),
						)
		pyg_dataset = self.to_pyg(data)
		loader = DataLoader(pyg_dataset, batch_size=1, shuffle=False)
		scores_all, y_all = [], []
		feature_names = np.array(data['markers'])
		labels_dict = {1:'Responder', 0:'Non-Responder'}
		for i, x_batch in enumerate(loader):
			x_batch = x_batch.to(self.device)
			kwargs = {'edge_weight':x_batch.edge_attr, 'batch': x_batch.batch}
			explanation = explainer(x_batch.x, x_batch.edge_index, index=None, **kwargs)
			scores, labels = explanation.visualize_feature_importance()
			scores = scores.cpu().numpy()
			sorted_indices = scores.argsort()[::-1]
			sorted_indices = sorted_indices[:10]
			sorted_scores = scores[sorted_indices]
			sorted_feature_names = feature_names[sorted_indices]
			sub_graph = explanation.get_explanation_subgraph()

			# import torch_geometric.utils as utils
			# coo_adj = utils.to_scipy_sparse_matrix(sub_graph.edge_index, edge_attr=sub_graph.edge_weight, num_nodes=sub_graph.node_mask.shape[0])
			# csr_adj = coo_adj.tocsr()
			# plt = visualise_cellgraph(csr_adj, random_state=42, node_labels=None, show=False)
			# buffer = io.BytesIO()
			# buffer.seek(0)
			# plt.savefig(buffer, format='png')
			# self.logger.log({f"results/SubGraph_{i}_{data['patient'][i]}_{labels_dict[x_batch.y.cpu().int().item()]}": wandb.Image(Image.open(buffer))})

			# plt.cla()
			# plt.clf()
			plt.figure(figsize=(16, 10))
			plt.bar(range(len(sorted_scores)), sorted_scores, tick_label=sorted_feature_names)
			plt.xlabel('Proteins', fontsize=28)
			plt.ylabel('Importance Score', fontsize=28)
			plt.title(f"GCN {labels_dict[x_batch.y.cpu().int().item()]} {data['patient'][i]}", fontsize=32)
			plt.xticks(rotation=45, ha='right', fontsize=28)
			plt.subplots_adjust(bottom=0.2)
			plt.tight_layout()

			buffer = io.BytesIO()
			buffer.seek(0)
			plt.savefig(buffer, format='png')
			self.logger.log({f"GNNExplainer ROI {i} {data['patient'][i]}": wandb.Image(Image.open(buffer))})
			scores_all.append(scores)
			y_all.append(x_batch.y)

		y_all = torch.stack(y_all).cpu().squeeze().numpy().astype('int')
		scores_all = np.stack(scores_all)

		mu_resp = scores_all[y_all].mean(0)
		std_resp = scores_all[y_all].std(0)
		mu_noresp = scores_all[1 - y_all].mean(0)
		std_noresp = scores_all[1 - y_all].std(0)

		feature_names = data['markers']

		df = pd.DataFrame({'Feature': np.array(feature_names), 
									'Mean Score pCR': mu_resp,
									'Std Score pCR': std_resp,
									'Mean Score Non-Responder': mu_noresp,
									'Std Score Non-Responder': std_noresp,
								})
		df.set_index('Feature', inplace=True)
		fig = plt.figure(figsize=(16, 10))
		x = np.arange(len(feature_names))
		bar_width = 0.35
		plt.bar(x - bar_width/2, df['Mean Score pCR'], yerr=df['Std Score pCR'], width=bar_width, label='Responder', capsize=5, alpha=0.7)
		plt.bar(x + bar_width/2, df['Mean Score pCR'], yerr=df['Std Score Non-Responder'], width=bar_width, label='Non-Responder', capsize=5, alpha=0.7)
		plt.ylabel('GNNExplainer Scores Averaged Across ROIs', fontsize=28)
		plt.title('GNNExplainer Scores Averaged Across ROIs', fontsize=32)
		plt.legend(fontsize=12, prop={'size': 8})
		plt.xticks(ticks=np.arange(len(feature_names)), labels=df.index, rotation=45, ha='right')
		plt.tight_layout()

		buffer = io.BytesIO()
		buffer.seek(0)
		plt.savefig(buffer, format='png')
		self.logger.log({'Average GNNExplainer across ROIs': wandb.Image(Image.open(buffer))})

	def gradient_attribution(self, data):
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
			logits = self.model(x_batch.x, x_batch.edge_index, x_batch.edge_attr, x_batch.batch)
			logits[0].backward()
			y_all.append(x_batch.y)
			node_gradients = x_batch.x.grad

			avg_node_gradients.append(node_gradients.abs().mean(0))
			std_node_gradients.append(node_gradients.abs().std(0))

			node_gradients = node_gradients.T.abs().cpu().numpy() 
			top_k_idx = node_gradients.argsort(1)[:,-100:]
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





