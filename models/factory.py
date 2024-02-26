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
	def __init__(self, input_dim, 
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
	def __init__(self, config,
						logger=None):
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
		self.model.eval()
		dataset = self.to_pyg(X, graphs)
		loader = DataLoader(dataset, batch_size=self.config['batch_size'], shuffle=False)
		preds = np.array([])
		for x_batch in loader:
			x_batch = x_batch.to(self.device)
			pred_i = self.model(x_batch.x, x_batch.edge_index, x_batch.edge_weight, x_batch.batch)
			pred_i = (pred_i>0.5).astype(float)
			preds = preds.concatenate(preds_i.cpu().numpy())
		return preds

	def predict_proba(self, X, graphs):
		self.model.eval()
		dataset = self.to_pyg(X, graphs)
		loader = DataLoader(dataset, batch_size=self.config['batch_size'], shuffle=False)
		preds = np.array([])
		for x_batch in loader:
			x_batch = x_batch.to(self.device)
			score = self.model(x_batch.x, x_batch.edge_index, x_batch.edge_weight, x_batch.batch)
			preds = score.cpu().numpy()
			preds = preds.concatenate(preds_i)

			preds.append(torch.exp(score[preds]))
		return preds

	def to_pyg(self, X, graphs, labels=None):
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