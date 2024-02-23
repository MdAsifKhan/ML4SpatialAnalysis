from sklearn.linear_model import LogisticRegression
import torch.nn as nn
import torch
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data

class GCN(nn.Module):
	def __init__(self, input_dim, hidden_dim, output_dim):
		super(GCN, self).__init__()		
		self.conv1 = GCNConv(input_dim, hidden_dim)
		self.conv2 = GCNConv(input_dim, output_dim)

	def forward(self, x, edge_index, edge_weight):
		x = self.conv1(x, edge_index, edge_weight)
		x = F.relu(x)
		x = self.conv2(x, edge_index, edge_weight)
		return F.log_softmax(x, dim=1)

class GCNmodel(nn.Module):
	def __init__(self, config):
		super(GCNmodel, self).__init__()
		self.config = config
		self.model = GCN(self.config['input_dim'], 
							self.config['hidden_dim'],
							self.config['output_dim'])
		self.optim = torch.optim.Adam(self.model.parameters(), 
										lr=self.config['lr'])

	def fit(self, X, y, graphs):
		X, G, y = selg.to_pyg(X, y, graphs)
		num_samples = X.shape[0]

		for epoch in range(self.config['nm_epochs']):
			for i in range(num_samples):
				loss = F.nll_loss(self.model(X[i], G[i].edge_index, G[i].edge_weight), y[i])
				loss.backward()
				self.optim.step()
			return None


	def to_pyg(self, X, graphs, y):
		X_tensor, G_tensor = [], []
		for node_attributes, graph in zip(X, graphs):
			X_tensor.append(node_attributes)
			row, col, data = graph.data, graph.indices, graph.values
			edge_index = torch.tensor((row, col)).long()
			edge_attr = torch.tensor(data).float()
			G = Data(edge_index=edge_index, edge_attr=edge_attr)
			G_tensor.append(G)
		y = torch.tensor(y)
		return X_tensor, G_tensor, y

