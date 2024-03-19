import torch
import torch.optim as optim
import torch.nn as nn
from captum.attr import IntegratedGradients

from imagemodels.models import ResNetClassifier
from mainutils.utils import compute_scores
MODELS = {
	"ResNetClassifier": ResNetClassifier,
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
			self.evaluate(train_loader, mode='train')
			self.logger.log({'Train Epoch Loss': batch_loss/len(train_loader)})

			if (epoch + 1) % self.config['test_every'] == 0:
				self.test(test_loader)
				self.evaluate(test_loader, mode='test')

	def evaluate(self, data_loader, mode='train'):
		with torch.no_grad():
			all_predictions, all_labels = [], []
			for images, labels in data_loader:
				images = images.to(self.config['device'])
				labels = labels.to(self.config['device'])

				outputs = self.classifier(images)
				predictions = torch.round(torch.sigmoid(outputs.squeeze())).cpu().numpy()

				all_predictions.append(predictions)
				all_labels.append(labels.squeeze().cpu().numpy())		
		metrics = compute_scores(all_predictions, all_labels, mode)
		metrics_table=[[key, value] for key, value in metrics.items()]
		self.logger.log({
					'Metrics': 
							wandb.Table(
									data=metrics_table, 
								columns=['Metric', 'Value'])
					})
		self.attribution(data_loader)


	def test(self, data_loader):
		with torch.no_grad():
			batch_loss = 0
			for batch_idx, (data, labels) in enumerate(data_loader):
				# Your training code here
				outputs = self.classifier(data).squeeze()
				loss = self.criterion(outputs, labels)
				self.logger.log({f"Iteration": batch_idx, 'Test Loss': loss.item()})
				batch_loss += loss.item() 
			self.logger.log({'Test Epoch Loss': batch_loss/len(data_loader)})


	def attribution(self, test_loader):
		sample_image, sample_label = next(iter(test_loader))
		sample_image = sample_image.to(self.config['device'])
		unique_labels = {1: 'Responder', 0: 'Non-Responder'}
		# Compute saliency map
		saliency_map = IntegratedGradients(self.classifier)
		attributions = saliency.attribute(sample_image, target=sample_label)
		# Log saliency map to WandB
		wandb.log({f"Saliency Map {sample_label}": [wandb.Image(saliency_map.cpu().detach().numpy())]})

		# Compute integrated gradients
		integrated_gradients = IntegratedGradients(self.classifier)
		attributions_ig, delta = integrated_gradients.attribute(sample_image, target=sample_label, return_convergence_delta=True)
		# Log integrated gradients to WandB
		wandb.log({"Integrated Gradients": [wandb.Image(integrated_gradients.cpu().detach().numpy())]})



