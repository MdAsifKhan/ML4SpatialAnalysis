import argparse
import wandb
from torchvision import transforms
from torch.utils.data import DataLoader
from imagemodels.dataset import create_patient_split, IMCDataset
from imagemodels.trainer import ImageTrainer
from mainutils.utils import load_config
from torch_geometric.seed import seed_everything
import pickle

def run(config):
	seed_everything(config['seed'])
	logname = f"_model_{config['trainer']['model_name']}"\
				f"_pretrained_{config['trainer'][config['trainer']['model_name']]['pretrained']}"\
				f"_seed_{config['seed']}"

	logger = wandb.init(entity="tnbcspatialcell", project="ML on Image Data", config=config, name=logname)

	train_patients, test_patients = create_patient_split(config['dataset']['DATA_PATH'], 
														config['dataset']['img_foldername'],
														config['dataset']['test_ratio'], 
														config['seed'])

	patient_split = {'train': train_patients, 'test': test_patients}
	with open(f"{config['trainer']['LOG_PATH']}/patient_splits.pkl", 'wb') as f:
		pickle.dump(patient_split, f)

	# Define transformations
	transform_train = transforms.Compose([
				transforms.Resize((512, 512)),
				transforms.RandomRotation(degrees=20),  # Random rotation
				transforms.RandomHorizontalFlip(p=0.5),  # Random horizontal flip
				transforms.RandomVerticalFlip(p=0.5),  # Random vertical flip
				transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10),  # Random affine transformations
	])


	transform_test = transforms.Compose([
				transforms.Resize((512, 512)),
	])

	
	# Initialize dataset and data loader for train and test sets
	train_dataset = IMCDataset(config['dataset'], train_patients, transform=transform_train, mode='train')
	train_loader = DataLoader(train_dataset, batch_size=config['dataset']['batch_size'], shuffle=True)


	test_dataset = IMCDataset(config['dataset'], test_patients, transform=transform_test, mode='test')
	test_loader = DataLoader(test_dataset, batch_size=config['dataset']['batch_size'], shuffle=False)

	config['trainer'][config['trainer']['model_name']]['nm_markers'] = len(train_dataset.use_markers)

	trainer = ImageTrainer(config['trainer'], train_patients, test_patients, logger)

	if config['resume']:
		trainer.load_model(resume_epoch)

	print(f"Training Image Classification Model on Patients {train_patients}")
	trainer.optimise(train_loader, test_loader)

	print(f"Testing Image Classification Model on Patients {test_patients}")
	trainer.evaluate(test_loader, mode='test')

	wandb.finish()

# Entry point for the script, parses arguments and loads configuration
if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--config', type=str, default='configs/config_images.yaml', help='Configuration file')
	args = parser.parse_args()
	config_file = load_config(args.config)
	run(config_file)