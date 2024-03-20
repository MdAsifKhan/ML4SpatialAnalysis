import os
import random
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from collections import defaultdict
import pandas as pd

def create_patient_split(data_path, img_folder, test_ratio=0.3, random_state=42):
	leap_folders = os.listdir(f"{data_path}/{img_folder}")
	patients = []
	for roi in leap_folders:
		patient_id = roi.split('_')[0].lower()
		if 'leap' in patient_id:
			patients.append(patient_id)
	unique_patients = list(set(patients))
	random.seed(random_state)
	random.shuffle(unique_patients)
	test_idx = int(test_ratio*len(unique_patients))
	train_patients = unique_patients[:-test_idx]
	test_patients = unique_patients[-test_idx:]
	return train_patients, test_patients

class IMCDataset(Dataset):
	def __init__(self, config, patients, transform=None, mode='train'):
		self.config = config
		self.patients = patients
		self.transform = transform
		self.leap_folders = os.listdir(f"{self.config['DATA_PATH']}/{self.config['img_foldername']}")

		meta_data = pd.read_csv(f"{self.config['DATA_PATH']}/{self.config['response_filename']}.csv", sep=',')
		self.leap_to_label = dict(zip(meta_data['LEAP_ID'].str.lower(), meta_data['Response']))
		self.unique_labels = {'pCR': 1.0, 'Responder': 1.0, 'Non-Responder': 0.0}
		self.patient_data, self.labels = [], []
		for roi in self.leap_folders:
			patient_id = roi.split('_')[0].lower()
			if (patient_id in self.patients) and (patient_id in self.leap_to_label.keys()):
				label = self.leap_to_label[patient_id]
				if label in self.unique_labels:
					roi_image_folder = os.path.join(f"{self.config['DATA_PATH']}/{self.config['img_foldername']}", roi)
					if len(os.listdir(roi_image_folder)) == self.config['nm_markers']:
						self.patient_data.append(roi_image_folder)
						self.labels.append(self.unique_labels[label])

	def __len__(self):
		return len(self.patient_data)

	def __getitem__(self, idx):
		images = []
		for image in os.listdir(self.patient_data[idx]):
			if image.lower().endswith(('.tif', '.tiff')):
				image = Image.open(os.path.join(self.patient_data[idx], image))
				if self.transform:
					image = self.transform(image.convert('RGB'))
				images.append(image)
		if len(images) == 0:
			print(self.patient_data[idx])
		images = torch.stack(images)
		return images, torch.tensor(self.labels[idx], dtype=torch.float32)