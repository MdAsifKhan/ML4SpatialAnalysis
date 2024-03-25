import torch
import torch.optim as optim
import torch.nn as nn
from captum.attr import IntegratedGradients

from imagemodels.models import ResNetClassifier, VisionTransformer
from mainutils.utils import compute_scores
import numpy as np
import wandb

MODELS = {
	"ResNetClassifier": ResNetClassifier,
	"VisionTransformer": VisionTransformer,
}

class ImageTrainer:
	def __init__(self, config, train_patients, test_patients, logger):
		self.config = config
		self.logger = logger	
		self.train_patients = train_patients
		self.test_patients = test_patients

		self.classifier = MODELS[self.config['model_name']](self.config[self.config['model_name']]).to(self.config['device'])
		
		self.optimizer = optim.Adam(self.classifier.parameters(), lr=self.config['optim']['lr'])
		self.criterion = nn.BCEWithLogitsLoss()

	def optimise(self, train_loader, test_loader):
		for epoch in range(self.config['nm_epochs']):
			batch_loss = 0
			for batch_idx, (data, labels) in enumerate(train_loader):
				data, labels = data.to(self.config['device']), labels.to(self.config['device'])
				self.optimizer.zero_grad()
				outputs = self.classifier(data)
				loss = self.criterion(outputs.squeeze(), labels.squeeze())
				loss.backward()
				self.optimizer.step()
				self.logger.log({'Train Iteration Loss': loss.item()})
				batch_loss += loss.item()
			self.evaluate(train_loader, mode='Train')
			self.logger.log({'Train Epoch Loss': batch_loss/len(train_loader)})

			if (epoch + 1) % self.config['test_every'] == 0:
				self.log_test_loss(test_loader)
				self.evaluate(test_loader, mode='Test')
				#self.attribution(data_loader)
			if (epoch + 1) % self.config['save_every'] == 0:
				self.save_model(epoch+1)

	def evaluate(self, data_loader, mode='train'):
		self.classifier.eval()
		with torch.no_grad():
			all_predictions, all_labels = np.array([]), np.array([])
			i = 0
			for images, labels in data_loader:
				i += 1
				images = images.to(self.config['device'])
				labels = labels.to(self.config['device'])

				outputs = self.classifier(images)
				
				if len(outputs)>1:
					outputs = outputs.squeeze()
					labels = labels.squeeze()

				predictions = torch.round(torch.sigmoid(outputs)).cpu().numpy()
				all_predictions = np.concatenate([all_predictions.reshape(-1), predictions.reshape(-1)])
				all_labels = np.concatenate([all_labels, labels.cpu().numpy()])

		metrics = compute_scores(all_labels, all_predictions, mode)
		metrics_table=[[key, value] for key, value in metrics.items()]
		self.logger.log({
					f"{mode} Metrics": 
							wandb.Table(
									data=metrics_table, 
								columns=['Metric', 'Value'])
					})
		

	def log_test_loss(self, data_loader):
		with torch.no_grad():
			batch_loss = 0
			for batch_idx, (data, labels) in enumerate(data_loader):
				# Your training code here
				data = data.to(self.config['device'])
				labels = labels.to(self.config['device'])
				outputs = self.classifier(data)
				loss = self.criterion(outputs.squeeze(), labels.squeeze())
				self.logger.log({'Test Iteration Loss': loss.item()})
				batch_loss += loss.item() 
			self.logger.log({'Test Epoch Loss': batch_loss/len(data_loader)})


	def save_model(self, name):
		"""
		Save a PyTorch model to a file.

		Args:
		    name : The file path to save the model.
		"""
		ckpt = {
				'model': self.classifier.state_dict(),
				'optimizer': self.optim.state_dict()
		}
		filepath = f"{self.config['LOG_PATH']}/{self.config['model_name']}_{name}.pth"
		torch.save(ckpt, filepath)


	def load_model(self, name):
		"""
		Save a PyTorch model to a file.

		Args:
		    name : The file name to save the model.
		"""
		filepath = f"{self.config['LOG_PATH']}/{self.config['model_name']}_{name}.pth"
		ckpt = torch.load(filepath, map_location=self.config['device'])
		
		self.classifier.load_state_dict(ckpt['model'])
		self.optimizer.load_state_dict(ckpt['optim'])


	def attribution(self, test_loader):
		sample_image, sample_label = next(iter(test_loader))
		sample_image = sample_image.to(self.config['device'])
		sample_label = sample_label.to(self.config['device'])
		unique_labels = {1: 'Responder', 0: 'Non-Responder'}
		# Compute saliency map
		saliency_map = IntegratedGradients(self.classifier)
		attributions = saliency_map.attribute(sample_image, target=sample_label)
		import wandb
		wandb.finish()
		import pdb
		pdb.set_trace()
		# Log saliency map to WandB
		wandb.log({f"Saliency Map {sample_label}": [wandb.Image(saliency_map.cpu().detach().numpy())]})

		# Compute integrated gradients
		integrated_gradients = IntegratedGradients(self.classifier)
		attributions_ig, delta = integrated_gradients.attribute(sample_image, target=sample_label, return_convergence_delta=True)
		# Log integrated gradients to WandB
		wandb.log({"Integrated Gradients": [wandb.Image(integrated_gradients.cpu().detach().numpy())]})



