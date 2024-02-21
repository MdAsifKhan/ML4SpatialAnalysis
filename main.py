import argparse
from utils.utils import load_config
from models.factory import ModelTrainer
from datautils.dataset import SpatialCellToFeatures
from sklearn.model_selection import train_test_split
import pdb
import wandb
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score


def compute_metrics(y_train, y_pred_train, y_test, y_pred_test):
	accuracy_train = accuracy_score(y_train, y_pred_train)
	auc_train = roc_auc_score(y_train, y_pred_train)
	f1_train = f1_score(y_train, y_pred_train)

	accuracy_test = accuracy_score(y_test, y_pred_test)
	auc_test = roc_auc_score(y_test, y_pred_test)
	f1_test = f1_score(y_test, y_pred_test)
	metrics = {
			'Accuracy Train': accuracy_train,
			'Accuracy Test': accuracy_test,
			'AUC Train': auc_train,
			'AUC Test': auc_test,
			'F1 Score Train': f1_train,
			'F1 Score Test': f1_test,			
	}
	return metrics

# x_t = A x_t + B h_t
# y_t = C x_t + D h_t

def train(config):
	logname = f"{config['dataset']['gtype']}_graph_{config['dataset']['use_graph']}"
	logger = wandb.init(project=f"ML on TNBC Data", config=config, name=logname)
	print('Preparing Features')
	dataloader = SpatialCellToFeatures(config['dataset'])

	print('Configuring models')
	model = ModelTrainer(config['mlmodel'], config['seed'])

	labels = dataloader.label_vec
	features = dataloader.featurisation()
	print('Creating data splits for training')
	X_train, X_test, y_train, y_test = train_test_split(features, labels,
											test_size=config['testportion'],
											random_state=config['seed'])

	print('Fitting the Model')

	model.fit(X_train, y_train)

	y_pred_train = model.predict(X_train)
	y_pred_test = model.predict(X_test)

	metrics = compute_metrics(y_train, y_pred_train, y_test, y_pred_test)
	print(metrics)
	log_metrics(metrics, logger)
	wandb.finish()

def log_metrics(metrics, logger):
	metrics_table= [[key, value] for key, value in metrics.items()]
	logger.log({'Metrics': wandb.Table(data=metrics_table, columns=["Metric", "Value"])})

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/config.yaml', help='Configuration file')
    args = parser.parse_args()
    config = load_config(args.config)
    train(config)








