import argparse
import pandas as pd
import numpy as np
import wandb
from mainutils.utils import load_config
from models.trainer import ModelTrainer
from datautils.dataset import SpatialCellToFeatures

def log_features(features, labels, logger):
	"""
	Logs features and labels as a table in W&B for better visualization.

	Args:
		features (np.array): Array of features.
		labels (np.array): Array of labels.
		logger (wandb.Logger): W&B logger object.
	"""
	columns = [f"Feature_{i}" for i in range(features.shape[1])] + ["Label"]
	df = pd.DataFrame(np.column_stack((features, labels)), columns=columns)
	logger.log({"Feature_Matrix": wandb.Table(dataframe=df)})


def run(config):
	"""
	Runs the entire training process based on the provided configuration.

	Args:
		config (dict): Configuration dictionary containing training parameters.
	"""
	if config['dataset']['gtype'] == 'celltype':
		logname = f"{config['dataset']['gtype']}"\
					f"_fcriterion_{config['model']['fcriterion']}"\
					f"_graphfeats_{config['model']['gcriterion']}"\
					f"_eval_{config['model']['eval']}"
	else:
		logname = f"{config['dataset']['gtype']}"\
					f"_graphfeats_{config['model']['gcriterion']}"\
					f"_eval_{config['model']['eval']}"

	# Initialize W&B logger with project name, entity, configuration, and log name
	logger = wandb.init(entity="tnbcspatialcell", project="ML on TNBC Data", config=config, name=logname)
	print('Preparing Features')
	dataset = SpatialCellToFeatures(config['dataset'])
	print('Feature Class Labels')
	print(dataset.unique_labels)
	data = dataset.data
	config['model']['feature_dim'] = len(data['markers'])
	config['model']['gtype'] = config['dataset']['gtype']
	print('Configuring models')
	# Configure the model trainer
	model = ModelTrainer(config['model'], 
							logger=logger,
							logfile=logname)
	
	# Train the model and get evaluation metrics
	model.optimise(data)
	
	#model.log_coefficients(logger)

	# Log features as a table (commented out, uncomment if needed)
	# log_features(features, labels, logger)
	wandb.finish()


# Entry point for the script, parses arguments and loads configuration
if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--config', type=str, default='configs/config.yaml', help='Configuration file')
	args = parser.parse_args()
	config_file = load_config(args.config)
	run(config_file)