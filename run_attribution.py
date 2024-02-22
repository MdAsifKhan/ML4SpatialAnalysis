import argparse
from utils.utils import load_config
from models.trainer import ModelTrainer
from datautils.dataset import SpatialCellToFeatures
import pdb
import wandb
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

def log_metrics(metrics, logger):
	metrics_table= [[key, value] for key, value in metrics.items()]
	logger.log({'Metrics': wandb.Table(data=metrics_table, columns=["Metric", "Value"])})

def log_features(X, y, logger):
	columns = [f"Feature_{i}" for i in range(X.shape[1])] + ["Label"]
	df = pd.DataFrame(np.column_stack((X, y)), columns=columns)
	logger.log({"Feature_Matrix": wandb.Table(dataframe=df)})

def attribution(config):
	if config['dataset']['gtype'] == 'celltype':
		logname = f"{config['dataset']['gtype']}_fcriterion_{config['dataset']['fcriterion']}_usegraphfeatures_{config['dataset']['use_graph']}"
	else:
		logname = f"{config['dataset']['gtype']}_usegraphfeatures_{config['dataset']['use_graph']}"

	logger = wandb.init(project=f"ML on TNBC Data", config=config, name=logname, resume=True)
	print('Preparing Features')
	dataloader = SpatialCellToFeatures(config['dataset'])
	labels = dataloader.label_vec
	features = dataloader.featurisation()

	print('Configuring models')
	explainer = ModelAttribution(config['attribution'])
	explainer.log_coefficients(logname)
	wandb.finish()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/config.yaml', help='Configuration file')
    args = parser.parse_args()
    config = load_config(args.config)
    attribution(config)








