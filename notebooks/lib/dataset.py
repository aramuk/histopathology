####################################################
# This module contains objects associated with the #
# PCam dataset used in this project.               #
####################################################

import os

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset

class PCam(Dataset):
    """The Patch Camelyon (PCam) dataset [1].
    
    Retrieved from https://www.kaggle.com/c/histopathologic-cancer-detection/.

    [1] B. S. Veeling, J. Linmans, J. Winkens, T. Cohen, M. Welling. "Rotation 
        Equivariant CNNs for Digital Pathology". arXiv:1806.03962
    """

    def __init__(self, image_dir, csv_path, transform=None):
        """Create a PyTorch dataset of (image, label) pairs from PCam.

        Args:
            image_dir: Folder with image data in file system.
            csv_path: CSV file with image labels.
            transform: Transforms to apply before loading.
        """
        self.labels_df = pd.read_csv(csv_path)
        self.image_dir = image_dir
        self.transform = transform
        self.num_samples = len(self.labels_df.index)

    def __len__(self):
        """Returns the length of the PCam dataset."""
        return self.num_samples

    def __getitem__(self, idx):
        """Get the (image, label) pair at a given index in the PCam dataset."""
        if torch.is_tensor(idx):
            idx = idx.to_list()
        image_id = self.labels_df.iloc[idx, 0]
        image_path = os.path.join(self.image_dir, image_id + '.tif')
        image = Image.open(image_path).convert('RGB')
        label = self.labels_df.iloc[idx, 1]
        label = np.array([label]).astype('int')
        if self.transform:
            image = self.transform(image)
        return (image, label)

class UnlabeledPCam(Dataset):
    """The Patch Camelyon (PCam) dataset, without ground truth labels [1].
    
    Retrieved from https://www.kaggle.com/c/histopathologic-cancer-detection/.

    [1] B. S. Veeling, J. Linmans, J. Winkens, T. Cohen, M. Welling. "Rotation 
        Equivariant CNNs for Digital Pathology". arXiv:1806.03962
    """

    def __init__(self, image_dir, transform=None):
        """Create a PyTorch dataset of images from PCam.

        Args:
            image_dir: Folder with image data in file system.
            transform: Transforms to apply before loading.
        """
        if not os.path.exists(image_dir) or not os.path.isdir(image_dir):
            raise ValueError(f'Proposed image directory {image_dir} is not on this file system.')
        self.image_dir = image_dir
        self.transform = transform
        self.image_paths = os.listdir(self.image_dir)
        self.num_samples = len(self.image_paths)

    def __len__(self):
        """Returns the length of the unlabeled PCam dataset."""
        return self.num_samples

    def __getitem__(self, idx):
        """Get the image at a given index in the PCam dataset."""
        if torch.is_tensor(idx):
            idx = idx.to_list()
        image_path = os.path.join(self.image_dir, self.image_paths[idx])
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image