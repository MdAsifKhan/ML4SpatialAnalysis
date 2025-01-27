import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATv2Conv, TransformerConv, GINConv, GlobalAttention

from torch_geometric.nn import global_mean_pool, global_add_pool, global_max_pool
from torch_geometric.nn import TopKPooling, SAGPooling, EdgePooling
from torch_geometric.nn import MLP


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
		self.conv1 = GCNConv(self.input_dim, 2*self.hidden_dim)
		self.conv2 = GCNConv(2*self.hidden_dim, 4*self.hidden_dim)
		self.clf = nn.Linear(4*self.hidden_dim, self.nm_class)
		self.dropout = nn.Dropout(0.3)

	def hidden_representation(self, x, edge_index, edge_weight, batch):
		x = self.conv1(x, edge_index, edge_weight)
		x = self.dropout(F.relu(x))
		x = self.conv2(x, edge_index, edge_weight)
		x = F.relu(x)
		x = global_mean_pool(x, batch)
		return self.clf(x), x

	def forward(self, x, edge_index, edge_weight, batch):
		x = self.conv1(x, edge_index, edge_weight)
		x = self.dropout(F.relu(x))
		x = self.conv2(x, edge_index, edge_weight)
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


class GCNWithAttention(nn.Module):
	def __init__(self, 
		input_dim=35, 
		hidden_dim=64, 
		nm_class=2
		):
		super().__init__()
		self.conv1 = GCNConv(self.input_dim, 2*self.hidden_dim)
		self.conv2 = GCNConv(2*self.hidden_dim, 4*self.hidden_dim)

		# Suppose we want a simple gate = linear -> 1
		self.gate_nn = nn.Linear(hidden_dim, 1)
		self.att_pool = GlobalAttention(self.gate_nn)  # or pass an nn_module if needed

		self.clf = nn.Linear(4*self.hidden_dim, self.nm_class)

	def forward(self, x, edge_index, batch):
		# Basic GCN layers
		x = self.conv1(x, edge_index)
		x = F.relu(x)
		x = self.conv2(x, edge_index)
		x = F.relu(x)

		# GlobalAttention pooling
		graph_emb = self.att_pool(x, batch)  # shape = [num_graphs, hidden_dim]
		logits = self.clf(graph_emb)          # shape = [num_graphs, out_dim]
		return logits

	def hidden_representation(self, x, edge_index, edge_weight, batch):
		# Basic GCN layers
		x = self.conv1(x, edge_index)
		x = F.relu(x)
		x = self.conv2(x, edge_index)
		x = F.relu(x)

		# GlobalAttention pooling
		x = self.att_pool(x, batch)  # shape = [num_graphs, hidden_dim]
		return self.clf(x), x


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


class SAGEAttentionNet(nn.Module):
	def __init__(self, 
		input_dim=35, 
		hidden_dim=64, 
		nm_class=2, 
		drop_p=0.3):
		super().__init__()
		# Use SAGEConv or GINConv
		self.conv1 = SAGEConv(input_dim, hidden_dim)
		self.conv2 = SAGEConv(hidden_dim, hidden_dim*2)

		# Attention pooling: requires a gate network
		self.gate_nn = nn.Sequential(
			nn.Linear(hidden_dim*2, 1)  # or hidden_dim*2 -> hidden_dim -> 1
		)
		self.pool = GlobalAttention(self.gate_nn)  # Each node gets a gate score

		self.clf = nn.Linear(hidden_dim*2, nm_class)
		self.dropout = nn.Dropout(drop_p)

	def forward(self, x, edge_index, edge_weight, batch):
		x = self.conv1(x, edge_index)
		x = F.relu(x)
		x = self.dropout(x)

		x = self.conv2(x, edge_index)
		x = F.relu(x)

		# Attention-based global pooling
		# GlobalAttention expects x shape [num_nodes, hidden_dim], plus optional batch
		x = self.pool(x, batch)  # shape [batch_size, hidden_dim*2]

		return self.clf(x)

	def hidden_representation(self, x, edge_index, edge_weight, batch):
		x = self.conv1(x, edge_index)
		x = F.relu(x)
		x = self.dropout(x)

		x = self.conv2(x, edge_index)
		x = F.relu(x)

		x = self.pool(x, batch)
		return self.clf(x), x


def scatter_sum(src: torch.Tensor, index: torch.Tensor, dim_size: int = None) -> torch.Tensor:
	"""
	Summation-based grouping: out[index[i]] += src[i].
	Mimics torch_scatter.scatter(src, index, reduce='sum') in pure PyTorch.

	Args:
	src: [N, F] tensor of values to sum
	index: [N] int tensor, group index for each row in src
	dim_size: optional number of groups. If not provided, 
	  we take max(index)+1

	Returns:
	out: [num_groups, F], where num_groups = dim_size or index.max()+1
	"""
	if dim_size is None:
		dim_size = int(index.max().item()) + 1

	out = src.new_zeros(dim_size, src.size(1))  # [num_groups, F]
	out.index_add_(0, index, src)
	return out

class GlobalAttention(nn.Module):
	"""
	A custom GlobalAttention layer without torch_scatter. 
	We replicate gating-based attention pooling:
	- gate_nn: a small network mapping x -> gate of shape [N,1]
	- alpha = sigmoid(gate)
	- Weighted node embeddings = alpha * x
	- Sum over nodes by their batch assignment => graph-level embedding
	"""
	def __init__(self, gate_nn: nn.Module, nn_module: nn.Module = None):
		"""
		Args:
		gate_nn: A module mapping node embeddings -> [N,1].
		nn_module: Optional module to transform x before gating (like an MLP).
		"""
		super().__init__()
		self.gate_nn = gate_nn
		self.nn_module = nn_module

	def forward(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
		"""
		Args:
		x: Node embeddings, shape [N, D].
		batch: [N] int tensor, which graph each node belongs to.
		Returns:
		pooled: [B, D], one embedding per graph.
		"""
		# Optional transform of x
		if self.nn_module is not None:
			x = self.nn_module(x)

		# 1) Compute gate for each node
		gate = self.gate_nn(x)  # shape [N,1]
		alpha = torch.sigmoid(gate)  # shape [N,1] in [0,1]

		# 2) Weighted node embeddings
		weighted_x = alpha * x  # broadcast => [N, D]

		# 3) Summation by 'batch'
		# We replicate scatter_sum logic
		# Need a dimension size => number of graphs in batch
		num_graphs = int(batch.max().item()) + 1
		pooled = scatter_sum(weighted_x, batch, dim_size=num_graphs)  # [B, D]

		return pooled

