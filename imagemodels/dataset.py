import os
import random
from PIL import Image
import torch
from torch.utils.data import Dataset
from collections import defaultdict
import pandas as pd
import tifffile as tp
import numpy as np
from torchvision.transforms.functional import resize

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
		self.all_markers = ['Alpha-SMA', 'B7-H4', 'Beta-Catenin', 'CD107a', 'CD11b', 'CD14', 'CD16', 
							'CD163', 'CD20', 'CD27', 'CD3', 'CD31', 'CD366', 'CD38', 'CD4', 'CD44', 
							'CD45', 'CD45RO', 'CD68', 'CD8a', 'Collage-Type_I', 'Carboplatin',
							'DNA1', 'DNA2', 'E-Cadherin', 'EGFR', 'FOXP3', 'Granzyme-B', 'HLA-DR-DQ-DP', 
							'Ki-67', 'PD-1', 'PD-L1', 'PD-L2', 'Pan-keratin', 'Tbet', 'VEGF', 
							'Vimentin', 'p53']

		self.exclude_markers = ['Carboplatin']
		self.use_markers = list(filter(lambda x: x not in self.exclude_markers, self.all_markers))
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
					markers = os.listdir(roi_image_folder)
					markers = [marker[:-5] for marker in markers if marker[:-5] not in self.exclude_markers]
					if sorted(set(markers)) == sorted(self.use_markers):
						self.patient_data.append(roi_image_folder)
						self.labels.append(self.unique_labels[label])


	def __len__(self):
		return len(self.patient_data)

	def __getitem__(self, idx):
		files = os.listdir(self.patient_data[idx])
		files.sort()
		images, channel_names = [], []
		for image in files:
			if image.lower().endswith(('.tif', '.tiff')) and (image[:-5] not in self.exclude_markers):
				image = tp.imread(os.path.join(self.patient_data[idx], image)).astype('float32')
				image = torch.tensor(np.arctan(image))
				if self.transform:
					image = self.transform(image.unsqueeze(0))
				channel_names.append(image[:-5])

				images.append(image.squeeze())
		images = torch.stack(images)
		return images, torch.tensor(self.labels[idx], dtype=torch.float32), channel_names

