import argparse
from mainutils.utils import load_config
from models.evaluation import ModelAttribution
from datautils.dataset import SpatialCellToFeatures
import pdb
import wandb
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

def attribution(config):
	"""
	Performs model attribution and logs results to Weights & Biases.

	Args:
		config (dict): Configuration dictionary containing model and dataset settings.
	"""
	if config['dataset']['gtype'] == 'celltype':
		logname = f"{config['dataset']['gtype']}_fcriterion_{config['dataset']['fcriterion']}_usegraphfeatures_{config['dataset']['use_graph']}"
	else:
		logname = f"{config['dataset']['gtype']}_usegraphfeatures_{config['dataset']['use_graph']}"

	logger = wandb.init(project=f"ML on TNBC Data", config=config, name=logname, resume=True)
	print('Preparing Features')
	dataset = SpatialCellToFeatures(config['dataset'])
	print('Feature Class Labels')
	print(dataset.unique_labels)
	print('Configuring models')
	explainer = ModelAttribution(config)
	explainer.run_attribution(dataset.data_test)
	wandb.finish()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/config.yaml', help='Configuration file')
    args = parser.parse_args()
    config = load_config(args.config)
    attribution(config)