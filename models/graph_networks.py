import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATv2Conv, TransformerConv, GINConv, GlobalAttention

from torch_geometric.nn import global_mean_pool, global_add_pool, global_max_pool
from torch_geometric.nn import TopKPooling, SAGPooling, EdgePooling
from torch_geometric.nn import MLP
from torch_geometric.nn import GraphNorm, BatchNorm

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
					nm_class,
					drop_p=0.3):
		super(GCN, self).__init__()
		self.input_dim = input_dim
		self.hidden_dim = hidden_dim
		self.nm_class = nm_class
		self.conv1 = GCNConv(self.input_dim, self.hidden_dim)
		self.conv2 = GCNConv(self.hidden_dim, self.hidden_dim)
		self.conv3 = GCNConv(self.hidden_dim, self.hidden_dim)
		
		self.bn1 = BatchNorm(self.hidden_dim)
		self.bn2 = BatchNorm(self.hidden_dim)

		self.clf = nn.Sequential(
						nn.Linear(self.hidden_dim, self.hidden_dim //2),
						nn.ReLU(),
						nn.Dropout(0.5),
						nn.Linear(self.hidden_dim, 1)
					)
		self.dropout1 = nn.Dropout(0.2)
		self.dropout2 = nn.Dropout(0.2)

	def hidden_representation(self, x, edge_index, edge_weight, batch):
		x = self.conv1(x, edge_index, edge_weight)
		x = self.bn1(x)
		x = self.dropout1(F.relu(x))
		
		x = self.conv2(x, edge_index, edge_weight)
		x = self.bn2(x)
		x = self.dropout2(F.relu(x))

		x = self.conv3(x, edge_index, edge_weight)
		x = global_mean_pool(x, batch)
		return self.clf(x), x

	def forward(self, x, edge_index, edge_weight, batch):
		x = self.conv1(x, edge_index, edge_weight)
		x = self.bn1(x)
		x = self.dropout1(F.relu(x))
		
		x = self.conv2(x, edge_index, edge_weight)
		x = self.bn2(x)
		x = self.dropout2(F.relu(x))

		x = self.conv3(x, edge_index, edge_weight)
		x = global_mean_pool(x, batch)
		
		return self.clf(x)

	def induced_subgraph(self,  x, edge_index, edge_weight, batch):
		x = self.conv1(x, edge_index, edge_weight)
		x = self.dropout(F.relu(x))
		x = self.conv2(x, edge_index, edge_weight)
		x_pooled, perm = global_sort_pool(x, batch, k=50, return_perm=True)
		mask_0 = torch.eq(edge_index[0].unsqueeze(1), topk_node_indices).any(dim=1)  # Check for source node
		mask_1 = torch.eq(edge_index[1].unsqueeze(1), topk_node_indices).any(dim=1)  # Check for target node

		# Create a mask for edges where both nodes are in the top k
		mask = mask_0 & mask_1

		# Select only the edges that connect top k nodes
		edge_index_subgraph = edge_index[:, mask]
		return edge_index_subgraph


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
					nm_class=2,
					drop_p=0.3):
		super(SSGCN, self).__init__()
		self.input_dim = input_dim
		self.hidden_dim = hidden_dim
		self.nm_class = nm_class
		self.K = int(K)
		self.alpha = alpha
		self.conv1 = SSGConv(self.input_dim, self.hidden_dim, self.alpha, self.K)
		self.conv2 = SSGConv(self.hidden_dim, self.hidden_dim, self.alpha, self.K)
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



