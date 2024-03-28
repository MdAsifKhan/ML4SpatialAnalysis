import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression
import wandb
from models.abstract import AbstractModel


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

	def load_model(self, filename, fold=None):
		if fold:
			name = f"{self.config['name']}_{fold}"
		else:
			name = self.config['name']

		filename = f"{self.config['LOG_PATH']}/{name}_{filename}.pkl"

		with open(filename, 'rb') as f:
			load = pickle.load(f)
		self.classifier = load['model']
		self.scaler = load['scaler']

	def run(self, data, logname):
		"""
		Logs the coefficients of the logistic regression model to W&B.

		Args:
			logger (wandb.Logger): W&B logger object.
		"""
		self.load_model(logname)
		if self.config['eval'] == 'split':
			filename = f"{self.config['LOG_PATH']}/{logname}.pkl"
			self.evaluate(data, mode='Test')

		elif self.config['eval'] == 'LeaveOneOut':
			for i, (train_i, test) in enumerate(self.leave_one_out_split(data)):
				filename = f"{self.config['LOG_PATH']}/leaveoneout_{i+1}_patient_{test_set['patient'][0]}"
				self.load_model(filename)

				y_pred_train_i = self.predict(train_i)
				y_pred_test_i = self.predict(test_i)

				y_train = np.concatenate([y_train, train_i['labels']])
				y_test = np.concatenate([y_test, test_i['labels']])
				y_pred_train = np.concatenate([y_pred_train, y_pred_train_i])
				y_pred_test = np.concatenate([y_pred_test, y_pred_test_i])

			print(f"Evaluating the {self.config['name']} Model")
			metrics = compute_scores_train(y_train, y_pred_train, y_test, y_pred_test)
			self.log_metrics(metrics, mode='LeaveOneOutROILevel')
			print('Metrics at ROI Level',metrics)

			metrics_train = patient_level_scores(y_train, y_pred_train,  data['patient'], mode='Train', pcriterion=self.config['pcriterion'])
			self.log_metrics(metrics_train, mode='LeaveOneOutPatientLevelTrain')
			print('Metrics at Patient Level', metrics_train)

			metrics_test = patient_level_scores(y_test, y_pred_test,  data['patient'], mode='Test', pcriterion=self.config['pcriterion'])
			self.log_metrics(metrics_test, mode='LeaveOneOutPatientLevelTest')
			print('Metrics at Patient Level', metrics_test)
		else:
			assert 0, f"{self.config['eval']} Evaluation not implemented"
