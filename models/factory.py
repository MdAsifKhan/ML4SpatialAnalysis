from sklearn.linear_model import LogisticRegression
import torch.nn as nn
import torch
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import torch.nn.functional as F
import numpy as np
import pdb

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


class tnbcGCN:
	"""
	Class for training GCN models on TNBC (Triple-Negative Breast Cancer) expression and spatial data.

	Args:
		config (dict): Configuration dictionary containing model and training parameters.
		logger (optional, Logger): Logger object for logging training information.
	"""
	def __init__(self, 
					config,
					logger=None):
		"""
		Initializes the TNBC GCN model and optimizer.

		Args:
			config (dict): Configuration dictionary containing odel and training parameters.
			logger (optional, Logger): Logger object for logging training information.
		"""
		self.config = config
		self.logger = logger
		self.device = torch.device(config['device'])
		self.model = GCN(self.config['input_dim'], 
							self.config['hidden_dim'],
							nm_class=1).to(self.device)
		self.optim = torch.optim.Adam(self.model.parameters(), 
										lr=self.config['lr'])
		self.criterion = nn.BCELoss()

	def fit(self, X, y, graphs):
		"""
		Trains the GCN model on the provided data.

		Args:
			X (list): List of gene expression data for each sample.
			y (list): List of labels (0 or 1) for each sample.
			graphs (list): List of spatial adjacency matrices representing connections between samples.
		"""
		self.model.train()
		dataset = self.to_pyg(X, graphs, y)
		loader = DataLoader(dataset, batch_size=self.config['batch_size'], shuffle=True)
		for epoch in range(self.config['nm_epochs']):
			loss_epoch = 0.
			for x_batch in loader:
				x_batch = x_batch.to(self.device)
				self.optim.zero_grad()
				logits = self.model(x_batch.x, x_batch.edge_index, x_batch.edge_weight, x_batch.batch)
				y_batch = x_batch.y.unsqueeze(1)
				loss = self.criterion(logits, y_batch)
				self.optim.step()
				self.logger.log({'GCN Iteration Loss': loss.item()})
				loss_epoch += loss.item()
			loss_epoch = loss_epoch/(len(loader))
			self.logger.log({'GCN Epoch Loss': loss.item()})

	def predict(self, X, graphs):
		"""
		Predicts class labels using the trained GCN model.

		Args:
			X (list): List of gene expression data for each sample.
			graphs (list): List of spatial adjacency matrices representing connections between samples.

		Returns:
			preds (np.array) : Array of predicted class labels (0 or 1) for each sample.
		"""
		self.model.eval()
		dataset = self.to_pyg(X, graphs)
		loader = DataLoader(dataset, batch_size=self.config['batch_size'], shuffle=False)
		preds = np.array([])
		for x_batch in loader:
			x_batch = x_batch.to(self.device)
			preds_i = self.model(x_batch.x, x_batch.edge_index, x_batch.edge_weight, x_batch.batch)
			preds_i = (preds_i.squeeze().detach().cpu().numpy()>0.5).astype(float)
			preds = np.concatenate([preds, preds_i])
		return preds

	def predict_proba(self, X, graphs):
		"""
		Predicts class probabilities using the trained GCN model.

		Args:
			X (list): List of gene expression data for each sample.
			graphs (list): List of spatial adjacency matrices representing connections between samples.

		Returns:
			preds (np.array) : Array of predicted probability for each sample.
		"""
		self.model.eval()
		dataset = self.to_pyg(X, graphs)
		loader = DataLoader(dataset, batch_size=self.config['batch_size'], shuffle=False)
		preds = np.array([])
		for x_batch in loader:
			x_batch = x_batch.to(self.device)
			score = self.model(x_batch.x, x_batch.edge_index, x_batch.edge_weight, x_batch.batch)
			preds_i = score.squeeze().detach().cpu().numpy()
			preds = np.concatenate([preds, preds_i])

		return preds

	def to_pyg(self, X, graphs, labels=None):
		"""
			Converts input data (gene expression, graphs, and optional labels) into PyTorch Geometric Data objects.

		Args:
			X (list): List of gene expression data for each sample.
			graphs (list): List of spatial adjacency matrices representing connections between samples.
			labels (optional, list): List of labels (0 or 1) for each sample.

		Returns:
			dataset (list): List of PyTorch Geometric Data objects representing the samples.
		"""
		dataset = []
		num_samples = len(X)
		for i in range(num_samples):
			graph_attributes = torch.tensor(X[i]).float()
			graph = graphs[i]
			graph = graph.tocoo()
			if not (labels is None):
				label = torch.tensor(labels[i]).float()
			else:
				label = None
			row, col, data = graph.row, graph.col, graph.data
			edge_index = torch.tensor((row, col)).long()
			edge_attr = torch.tensor(data).float()
			data = Data(x=graph_attributes, edge_index=edge_index, edge_attr=edge_attr, y=label)
			dataset.append(data)
		return dataset