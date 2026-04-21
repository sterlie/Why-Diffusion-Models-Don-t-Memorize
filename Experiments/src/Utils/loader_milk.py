from torchvision import transforms
import torch
from torch.utils.data import Dataset
import pandas as pd
import os
import numpy as np

class MILK10Dataset(Dataset):
	def __init__(self, metadata_csv, image_dir=None, image_pth=None, label_col='skin_tone_class', img_size=32, transform=None):
		"""
		Args:
			metadata_csv (str): Path to metadata CSV file.
			image_dir (str): Directory with images (if loading individual files).
			image_pth (str): Path to .pth file with all images (optional).
			label_col (str): Column name in CSV to use as label.
			transform (callable, optional): Optional transform to be applied on a sample.
		"""
		self.df = pd.read_csv(metadata_csv)
		self.image_dir = image_dir
		self.image_pth = image_pth
		self.label_col = label_col
		# Compose resize transform to 32x32 with any user-provided transform
		resize_transform = transforms.Resize((img_size, img_size))
		if transform is not None:
			self.transform = transforms.Compose([resize_transform, transform])
		else:
			self.transform = resize_transform

		# Get all unique labels for one-hot encoding
		self.label_values = sorted(self.df[label_col].unique())
		self.label_to_idx = {v: i for i, v in enumerate(self.label_values)}
		self.num_classes = len(self.label_values)

		# Optionally load all images from .pth file
		if image_pth is not None and os.path.exists(image_pth):
			self.images = torch.load(image_pth)
		else:
			self.images = None

	def __len__(self):
		return len(self.df)

	def __getitem__(self, idx):

		row = self.df.iloc[idx]
		isic_id = row['isic_id']

		# Load image
		if self.images is not None:
			# Assume images is a dict: isic_id -> image tensor
			image = self.images[isic_id]
		else:
			# Implement image loading from file if needed
			# Example: image_path = os.path.join(self.image_dir, f"{isic_id}.jpg")
			raise NotImplementedError("Image loading from files not implemented. Use .pth for now.")

		# Get label and one-hot encode
		label_value = row[self.label_col]
		label_idx = self.label_to_idx[label_value]
		label_onehot = np.zeros(self.num_classes, dtype=np.float32)
		label_onehot[label_idx] = 1.0
		label_onehot = torch.from_numpy(label_onehot)

		if self.transform:
			image = self.transform(image)

		return image, label_onehot

# Example usage:
# dataset = MILK10Dataset(
#     metadata_csv="/path/to/MILK10k_Training_Metadata.csv",
#     image_pth="/path/to/MILK10k_Training_Input.pth",
#     label_col="skin_tone_class"
# )
