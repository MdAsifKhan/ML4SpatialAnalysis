import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression
import wandb
from models.abstract import AbstractModel
from mainutils.utils import compute_scores_train, compute_scores
from mainutils.utils import leave_one_out_split, patient_level_scores

class ModelEvaluation(AbstractModel):
	"""
	This class performs model evaluation and attribution.
	Attributes:
		config (dict): Configuration dictionary containing training parameters.
		classifier (object): Trained logistic regression model.
		feature_names (list): List of feature names.
		clf_name (str): Name of the classifier type.
	"""
	def __init__(self, config, logname, logger=None):
		"""
		Initializes the ModelEvaluation object.

		Args:
			config (dict): Configuration dictionary containing attribution parameters.
		"""
		super().__init__(config, logger)
		self.classifier = None
		self.scaler = None

	def run(self, dataset, logname):
		"""
		Logs the coefficients of the logistic regression model to W&B.

		Args:
			logger (wandb.Logger): W&B logger object.
		"""
		if self.config['eval'] == 'split':
			filename = f"{self.config['LOG_PATH']}/{self.config['name']}_{logname}.pkl"
			self.load_model(logname)
			self.evaluate(dataset['test'], mode='Test')

		elif self.config['eval'] == 'leaveOneOut':
			y_test = np.array([])
			y_pred_test = np.array([])
			y_pred_test_prob = np.array([])
			patient_label = np.array([])
			for i, (train_i, test_i) in enumerate(leave_one_out_split(dataset)):
				filename = f"{self.config['LOG_PATH']}/{logname}_leaveoneout_{i+1}_patient_{test_i['patient'][0]}.pkl"
				self.load_model(filename)

				# y_pred_train_i = self.predict(train_i)
				# y_pred_train_prob_i = self.predict_proba(train_i)[:, 1]

				# print(f"Evaluating the {self.config['name']} Model on Training Set")
				# print(f"[Fold {i+1}] Evaluating {self.config['name']} Model on TRAIN set")
				# train_metrics = compute_scores(
				# 				train_i['labels'],          # ground truth
				# 				y_pred_train_i,             # predicted labels
				# 				y_pred_train_prob_i,        # predicted probabilities
				# 				mode='Train'
				# )
				# self.log_metrics(train_metrics, mode=f"LOO_ROI_Train_Fold{i+1}")

				# metrics_train = patient_level_scores(
				# 				train_i['labels'], 
				# 				y_pred_train_i, 
				# 				y_pred_train_prob_i, 
				# 				train_i['patient'], 
				# 				mode='Train', 
				# 				pcriterion=self.config['pcriterion'])
				# print('Metrics at Patient Level', metrics_train)
				# self.log_metrics(metrics_train, mode=f"LeaveOneOutPatientLevelTrain {i+1}")

				y_pred_test_i = self.predict(test_i)
				y_pred_test_prob_i = self.predict_proba(test_i)[:, 1] 

				y_test = np.concatenate([y_test, test_i['labels']])
				y_pred_test = np.concatenate([y_pred_test, y_pred_test_i])
				y_pred_test_prob = np.concatenate([y_pred_test_prob, y_pred_test_prob_i])
				patient_label = np.concatenate([patient_label, test_i['patient']])

			print(f"[All Folds] Evaluating {self.config['name']} Model on the concatenated TEST folds")

			metrics = compute_scores(y_test, y_pred_test, y_pred_test_prob, mode='Test')
			self.log_metrics(metrics, mode=f"LeaveOneOutROILevel {i+1}")

			metrics_test = patient_level_scores(y_test, y_pred_test, y_pred_test_prob, patient_label, mode='Test', pcriterion=self.config['pcriterion'])
			self.log_metrics(metrics_test, mode='LeaveOneOutPatientLevelTest')
			print('Metrics at Patient Level', metrics_test)

		else:
			assert 0, f"{self.config['eval']} Evaluation not implemented"

	def get_explanation(self, test_data, data, top_k_count_positive, top_k_count_negative):
		from torch_geometric.explain import Explainer, GNNExplainer
		self.model.eval()
		explainer = Explainer(
				model=self.classifier.model,
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
			self.logger.log({f"results/SubGraph_{data['leapid'][i]}_{data['patient'][i]}_{labels_dict[x_batch.y.cpu().int().item()]}": wandb.Image(Image.open(buffer))})

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
			self.logger.log({f"GNNExplainer ROI {data['leapid'][i]} {data['patient'][i]}": wandb.Image(Image.open(buffer))})
			plt.close()

		return top_k_count_positive, top_k_count_negative
