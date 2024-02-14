import argparse
from utils import load_config
from models import ModelTrainer
from dataset import SpatialCellToFeatures
from sklearn.model_selection import train_test_split
import pdb
import wandb


def train(config):
	logger = wandb.init(project=f"ML on TNBC Data", config=config)
	print('Preparing Features')
	dataloader = SpatialCellToFeatures(config['dataset'])

	print('Configuring models')
	model = ModelTrainer(config['mlmodel'])

	features = dataloader.featurisation()
	labels = dataloader.label_vec
	print('Creating data splits for training')
	X_train, X_test, y_train, y_test = train_test_split(features, labels,
											test_size=config['testportion'],
											random_state=config['mlmodel']['seed'])

	print('Fitting the Model')

	model.fit(X_train, y_train)

	y_probs = model.predict_proba(X_test)
	train_acc = model.accuracy(X_train, y_train)
	test_acc = model.accuracy(X_test, y_test)

	wandb.log({'Train Accuracy'}: train_acc)
	wandb.log({'Test Accuracy'}: test_acc)

	wandb.log('PRcurve': wandb.plots.precision_recall(y_test, y_probs, labels))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml', help='Configuration file')
    args = parser.parse_args()
    config = load_config(args.config)
    train(config)








