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

def create_patient_split(data_path, response_filename, test_ratio=0.3, random_state=42):
	meta_data = pd.read_csv(f"{data_path}/{response_filename}.csv", sep=',')
	patient_to_leap = {}
	for patient, leap in zip(meta_data['Patient'], meta_data['LEAP_ID'].str.lower()):
		if patient in patient_to_leap:
			patient_to_leap[patient].append(leap)
		else:
			patient_to_leap[patient] = [leap]

	unique_patients = list(patient_to_leap.keys())
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
		
		self.patient_to_leap = {}
		for patient, leap in zip(meta_data['Patient'], meta_data['LEAP_ID'].str.lower()):
			if patient in self.patient_to_leap:
				self.patient_to_leap[patient].append(leap)
			else:
				self.patient_to_leap[patient] = [leap]

		self.leap_to_label = dict(zip(meta_data['LEAP_ID'].str.lower(), meta_data['Response']))
		self.leap_to_patient = dict(zip(meta_data['LEAP_ID'].str.lower(), meta_data['Patient']))

		self.unique_labels = {'pCR': 1.0, 'Responder': 1.0, 'Non-Responder': 0.0}

		self.patient_data, self.labels, self.patient_id = [], [], []
		for leap in os.listdir(f"{self.config['DATA_PATH']}/{self.config['img_foldername']}"):
			#Check if Leap has a patient ID and a Response Label
			if (leap.split('_')[0].lower() in self.leap_to_patient.keys()) and (leap.split('_')[0].lower() in self.leap_to_label.keys()):
				label = self.leap_to_label[leap.split('_')[0].lower()]
				if label in self.unique_labels:
					roi_image_folder = os.path.join(f"{self.config['DATA_PATH']}/{self.config['img_foldername']}", leap)
					markers = os.listdir(roi_image_folder)
					markers = [marker[:-5] for marker in markers if marker[:-5] not in self.exclude_markers]
					if sorted(set(markers)) == sorted(self.use_markers):
						self.patient_data.append(roi_image_folder)
						self.labels.append(self.unique_labels[label])
						self.patient_id.append(self.leap_to_patient[leap.split('_')[0].lower()])


	def __len__(self):
		return len(self.patient_data)

	def __getitem__(self, idx):
		files = os.listdir(self.patient_data[idx])
		files.sort()
		images, markers = [], []
		for image in files:
			if image.lower().endswith(('.tif', '.tiff')) and (image[:-5] not in self.exclude_markers):
				image = tp.imread(os.path.join(self.patient_data[idx], image)).astype('float32')
				image = torch.tensor(np.arctan(image))
				if self.transform:
					image = self.transform(image.unsqueeze(0))
				markers.append(image[:-5])

				images.append(image.squeeze())
		images = torch.stack(images)
		data = {
				'images': images,
				'labels': torch.tensor(self.labels[idx], dtype=torch.float32),
				'markers': markers,
				'patients': self.patient_id[idx]

		}
		return data

