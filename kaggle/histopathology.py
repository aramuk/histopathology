# Helper code for Histopathologic Cancer Detection
import os
import time

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from skimage import io
from sklearn.metrics import confusion_matrix
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torchvision
from torchvision import transforms
from torchvision.models import vgg16


class Library:
    """Group imports for convenience."""
    pass

dataset = Library()
models = Library()
training = Library()
evaluation = Library()
transforms = Library()

####################################################
# This module contains objects associated with the #
# PCam dataset used in this project.               #
####################################################
class PCam(Dataset):
    """The Patch Camelyon (PCam) dataset [1].
    
    Retrieved from https://www.kaggle.com/c/histopathologic-cancer-detection/.

    [1] B. S. Veeling, J. Linmans, J. Winkens, T. Cohen, M. Welling. "Rotation 
        Equivariant CNNs for Digital Pathology". arXiv:1806.03962
    """

    def __init__(self, image_dir, csv_path, transform=None):
        """ 
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
        """Get the size of the PCam dataset."""
        return self.num_samples

    def __getitem__(self, idx):
        """Get the (image, label) at a given index in the PCam dataset."""
        if torch.is_tensor(idx):
            idx = idx.to_list()
        image_id = self.labels_df.iloc[idx, 0]
        image_path = os.path.join(self.image_dir, image_id + '.tif')
        image = io.imread(image_path)
        label = self.labels_df.iloc[idx, 1]
        label = np.array([label]).astype('int')
        if self.transform:
            image = self.transform(image)
        return (image, label)

dataset.PCam = PCam


##############################################################
# This module contains the definitions for all convolutional #
# neural networks models used in this project.               #
##############################################################
class Veggie16(nn.Module):
    """A model that adapts the VGG-16 architecture.

    This network applies transfer learning to learn the parameters
    of VGG-16, and freezes those layers of the model. The classification
    layer of the architecture is modified and will be retrained to 
    predict the desired number of output classes.
    """

    def __init__(self, num_classes):
        """Creates a Veggie16 network.

        Args:
            num_classes: The number of output classes to predict.
        """
        super(Veggie16, self).__init__()
        # Define the model's name for it's output files
        # Load a pre-trained VGG-16 model and turn off autograd
        # so its weights won't change.
        architecture = vgg16(pretrained=True)
        for layer in architecture.parameters():
            layer.requires_grad = False
        # Copy the convolutional layers of the model.
        self.features = architecture.features
        # Copy the average pooling layer of the model.
        self.avgpool = architecture.avgpool
        # Define a new block of fully-connected layers for the model.
        in_ftrs = architecture.classifier[0].in_features
        self.classifier = nn.Sequential(
            nn.Linear(in_features=in_ftrs, out_features=1024, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(in_features=1024, out_features=1024, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(in_features=1024, out_features=num_classes, bias=True)
        )

    def forward(self, x):
        """Does a forward pass on an image x."""
        out = self.features(x)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out

models.Veggie16 = Veggie16


####################################################
# This module contains code used to train a model. #
####################################################
class Trainer:
    """A wrapper that trains a model """

    def __init__(self, model, device, model_dir='/kaggle/working/'):
        """Initialize a Trainer.

        Args:
            model: A PyTorch model
            device: Device to train the model on.
            model_dir: Where to save the model's checkpoint files (default=/kaggle/working/)
        """
        self.model = model
        self.device = device
        if not os.path.exists(model_dir) or not os.path.isdir(model_dir):
            raise ValueError(f'Proposed model directory {model_dir} does not exist or is not a folder.')
        self.checkpoint_file = os.path.join(model_dir, model.__class__.__name__ + '_ckpt.pth')

    def train(self, train_loader, criterion, optimizer, num_epochs=25):
        """Trains a given model.

        Args:
            train_loader: A DataLoader to the training data set.
            criterion: Loss function for the model.
            optimizer: Optimization algorithm to be used.
            num_epochs: The number of iterations of the optimizer (default=25).

        Return: (float) total_loss over all epochs.
        """
        self.model.train()
        since = time.time()
        num_steps = len(train_loader)
        total_loss = 0
        for epoch in range(1, num_epochs+1):
            for i, (images, labels) in enumerate(train_loader, start=1):
                images = images.to(self.device)
                labels = labels.to(self.device)
                # Generate prediction and evaluate
                outputs = self.model(images)
                loss = criterion(outputs, labels.long().flatten())
                # Backpropagate loss and update weights
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # Compute running average of epoch loss
                total_loss += float(loss)
                # Print progress every 1000 batches
                if i % 1000 == 0:
                    print(f'Epoch [{epoch}/{num_epochs}], Step [{i}/{num_steps}], Loss: {loss.item():.6f}')

            self.save_checkpoint()
        # Print training time
        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        return total_loss

    def save_checkpoint(self):
        """Saves current weights to checkpoint file."""
        torch.save(self.model.state_dict(), self.checkpoint_file)    

    def load_checkpoint(self):
        """Loads the weights from last checkpoint."""
        if os.path.exists(self.checkpoint_file):
            self.model.load_state_dict(torch.load(self.checkpoint_file))

training.Trainer = Trainer


#######################################################
# This module contains code used to evaluate a model. # 
#######################################################
def evaluate(model, val_loader, device, criterion):
    """Evaluates a given model.

    Args:
        model: A PyTorch model.
        test_loader: A DataLoader to the evluation data set.

    Returns: A tuple of the (F1-Score, Accuracy, Loss) for the model.
    """
    net_loss = 0.0
    total = 0.0
    correct = 0.0
    y = []
    y_hat = []

    model.eval()
    with torch.no_grad():
        since = time.time()
        for i, (images, labels) in enumerate(val_loader):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            predictions = torch.argmax(outputs, dim=1)

            y.extend(labels)
            y_hat.extend(predictions)

            # Compute running average of loss
            net_loss = (net_loss * (i-1) + loss.item()) / i
            # Compute totals
            total += labels.size(0)
            correct += (predictions == labels).sum().item()


    accuracy = total / correct
    score = f1_score(y, y_hat)
    return score, accuracy, net_loss


def f1_score(y, y_hat):
    """Computes the f1_score for a given pair of predictions and ground truths."""
    tn, fp, fn, tp = confusion_matrix(y, y_hat).ravel()
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return 2 * precision * recall / (precision + recall)

evaluation.evaluate = evaluate
evaluation.f1_score = f1_score


#################################################
# Transforms applied to images in this project. #
#################################################
class ToNormalized(object):
    """Perform channel-wise normalization on an RGB image


    """
    def __call__(self, pic):
        """
        :param pic: (numpy.ndarray): Image to be normalized

        :return: normalized_image: (numpy.ndarray): Mean-normalized image
        """
        normalized_image = torch.div(torch.sub(pic, self.mean), self.std)
        return normalized_image

    def __init__(self, mean, std):
        self.mean = torch.Tensor(mean).reshape(3, 1, 1)
        self.std = torch.Tensor(std).reshape(3, 1, 1)


class ToClosed(object):
    """
    Perform morphological closing on an RGB image with a 2,2 kernel
    """
    def __call__(self, img):
        closed_img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, self.kernel)
        closed_img = Image.fromarray(np.uint8(closed_img))
        return closed_img

    def __init__(self):
        self.kernel = np.ones([2, 2], np.uint8)

transforms.ToNormalized = ToNormalized
transforms.ToClosed = ToClosed