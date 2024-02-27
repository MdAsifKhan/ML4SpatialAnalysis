import argparse
from utils.utils import load_config
from models.trainer import ModelTrainer
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

	print('Configuring models')
	explainer = ModelAttribution(config['attribution'])
	explainer.log_coefficients(logname)

	X = explainer.featurisation(dataset.data)
	explainer.log_shap_scores(X)
	wandb.finish()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/config.yaml', help='Configuration file')
    args = parser.parse_args()
    config = load_config(args.config)
    attribution(config)