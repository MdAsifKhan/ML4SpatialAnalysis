import argparse
from utils.utils import load_config
from models.trainer import ModelTrainer
from datautils.dataset import SpatialCellToFeatures
import pdb
import wandb


def log_features(X, y, logger):
	columns = [f"Feature_{i}" for i in range(X.shape[1])] + ["Label"]
	df = pd.DataFrame(np.column_stack((X, y)), columns=columns)
	logger.log({"Feature_Matrix": wandb.Table(dataframe=df)})


def run(config):
	if config['dataset']['gtype'] == 'celltype':
		logname = f"{config['dataset']['gtype']}"\
					f"_fcriterion_{config['model']['fcriterion']}"\
					f"_graphfeats_{config['model']['gcriterion']}"\
					f"_eval_{config['model']['eval']}"
	else:
		logname = f"{config['dataset']['gtype']}"\
					f"_graphfeats_{config['model']['gcriterion']}"\
					f"_eval_{config['model']['eval']}"

	logger = wandb.init(entity="tnbcspatialcell", project="ML on TNBC Data", config=config, name=logname)
	print('Preparing Features')
	dataloader = SpatialCellToFeatures(config['dataset'])
	labels = dataloader.label_vec
	features = dataloader.expressions

	print('Configuring models')
	model = ModelTrainer(config['model'], 
									logger=logger,
									feature_names=dataloader.feature_names,
									logfile=logname)
	metrics = model.optimise(features, labels, dataloader.graphs, config['dataset']['gtype'])
	print(metrics)
	log_metrics(metrics, logger)
	model.log_coefficients(logger)
	# log_features(features, labels, logger)
	wandb.finish()


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--config', type=str, default='configs/config.yaml', help='Configuration file')
	args = parser.parse_args()
	config = load_config(args.config)
	run(config)