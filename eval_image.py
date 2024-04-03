import argparse
import wandb
from torchvision import transforms
from torch.utils.data import DataLoader
from imagemodels.dataset import create_patient_split, IMCDataset
from imagemodels.trainer import ImageTrainer
from mainutils.utils import load_config
from torch_geometric.seed import seed_everything

def run(config):
	seed_everything(config['seed'])
	logname = f"_model_{config['trainer']['model_name']}"\
				f"_pretrained_{config['trainer'][config['trainer']['model_name']]['pretrained']}"\
				f"_seed_{config['seed']}"

	logger = wandb.init(entity="tnbcspatialcell", project="ML on Image Data", config=config, name=logname, reinit=True)

	with open(f"{config['trainer']['LOG_PATH']}/patient_splits.pkl", 'rb') as f:
		patient_split = pickle.load(f)

	transform_test = transforms.Compose([
				transforms.Resize((512, 512)),  # Resize to a fixed size
	])


	test_dataset = IMCDataset(config['dataset'], test_patients, transform=transform_test, mode='test')
	test_loader = DataLoader(test_dataset, batch_size=config['dataset']['batch_size'], shuffle=False)

	config['trainer'][config['trainer']['model_name']]['nm_markers'] = config['dataset']['nm_markers']

	trainer = ImageTrainer(config['trainer'], train_patients, test_patients, logger)

	print(f"Testing Image Classification Model on Patients {test_patients}")
	trainer.evaluate(test_loader, mode='Test')

	wandb.finish()

# Entry point for the script, parses arguments and loads configuration
if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--config', type=str, default='configs/config_images.yaml', help='Configuration file')
	args = parser.parse_args()
	config_file = load_config(args.config)
	run(config_file)