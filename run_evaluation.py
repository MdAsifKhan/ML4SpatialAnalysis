import argparse
from mainutils.utils import load_config
from models.evaluation import ModelEvaluation
from datautils.dataset import SpatialCellToFeatures
import pdb
import wandb
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

def run(config):
	"""
	Performs model attribution and logs results to Weights & Biases.

	Args:
		config (dict): Configuration dictionary containing model and dataset settings.
	"""

	if config['model']['name'] == 'gcn':
		config['model']['gcriterion'] = 'gcn'

	logname = f"_model_{config['model']['name']}"\
				f"_graphtype_{config['dataset']['gtype']}"\
				f"_fnorm_{config['model']['fnorm']}"\
				f"_graphfeats_{config['model']['gcriterion']}"\
				f"_eval_{config['dataset']['datasplit']}"\
				f"_seed_{config['seed']}"

	logger = wandb.init(project=f"ML on TNBC Data", config=config, name=logname, resume=True)
	print('Preparing Features')
	dataset = SpatialCellToFeatures(config['dataset'], random_state=config['seed'])
	print('Feature Class Labels')
	print(dataset.unique_labels)
	print('Configuring models')
	explainer = ModelEvaluation(config, logname, logger)
	explainer.run(dataset.data_test)
	wandb.finish()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/config.yaml', help='Configuration file')
    args = parser.parse_args()
    config = load_config(args.config)
    run(config)