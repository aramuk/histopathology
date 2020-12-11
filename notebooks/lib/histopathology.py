####################################################
# Helper code for Histopathologic Cancer Detection #
####################################################
import os
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import PIL
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision.models import vgg16, alexnet, resnet34


class Album:
    """Group imports for convenience."""
    pass

dataset = Album()
models = Album()
training = Album()
evaluation = Album()
transforms = Album()

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
        image = PIL.Image.open(image_path).convert('RGB')
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

    def __init__(self, image_dir, transforms=None):
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
        image_path = self.image_paths.iloc[idx, 0]
        image = PIL.Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image

dataset.PCam = PCam
dataset.TestDataset = TestDataset


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

    def __init__(self, pretrained=True, freeze_weights=True):
        """Creates a Veggie16 network.

        Args:
            pretrained: Model should load the weights from a pretrained VGG-16.
            freeze_weights: Model should freeze weights of the convolutional layers.
        """
        super(Veggie16, self).__init__()
        # Define the model's name for it's output files
        # Load a pre-trained VGG-16 model and turn off autograd
        # so its weights won't change.
        architecture = vgg16(pretrained=pretrained)
        if freeze_weights:
            for layer in architecture.parameters():
                layer.requires_grad = False
        # Copy the convolutional layers of the model.
        self.features = architecture.features
        # Copy the average pooling layer of the model.
        self.avgpool = architecture.avgpool
        # Redefine the classification block of VGG-16.
        # Use LeakyReLU units instead of ReLU units.
        # Output layer has 2 nodes only for the 2 classes in the PCam dataset.
        in_ftrs = architecture.classifier[0].in_features
        self.classifier = nn.Sequential(
            nn.Linear(in_features=in_ftrs, out_features=4096, bias=True),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(in_features=4096, out_features=4096, bias=True),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(in_features=4096, out_features=2, bias=True)
        )
        # Define a LogSoftmax layer for converting outputs to probabilities
        # Not needed in `forward()` because included in nn.CrossEntropyLoss
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        """Does a forward pass on an image x."""
        out = self.features(x)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out

class AlmondNet(nn.Module):
    """
    A model that adapts the AlexNet CNN
    """
    
    def __init__(self, pretrained=True, freeze_weights=True):
        super(AlmondNet, self).__init__()
        # Load the pretrained AlexNet model, 
        # remove autograd to keep weights from changind
        architecture = alexnet(pretrained=pretrained)
        if freeze_weights:
            for layer in architecture.parameters():
                layer.requires_grad = False
        # copy architecture features and avg pool layer
        self.features = architecture.features
        self.avgpool = architecture.avgpool
        in_ftrs = architecture.classifier[1].in_features
        self.classifier = nn.Sequential(
            nn.Linear(in_features=in_ftrs, out_features=2048, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(in_features=2048, out_features=2048, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(in_features=2048, out_features=2, bias=True)
        )
        # Define a LogSoftmax layer for converting outputs to probabilities
        # Not needed in `forward()` because included in nn.CrossEntropyLoss
        self.log_softmax = nn.LogSoftmax(dim=1)
    
    def forward(self, x):
        """Does a forward pass on an image x."""
        out = self.features(x)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out

class RaisinNet34(nn.Module):
    """A model that adapts the ResNet-34 architecture.

    This network applies transfer learning to learn the parameters
    of ResNet-34, and freezes those layers of the model. The classification
    layer of the architecture is modified and will be retrained to 
    predict the desired number of output classes.
    """

    def __init__(self, pretrained=True, freeze_weights=True):
        """Creates a RaisinNet34 network.

        Args:
            pretrained: Model should load the weights from a pretrained ResNet-34.
            freeze_weights: Model should freeze weights of the convolutional layers.
        """
        super(RaisinNet34, self).__init__()
        # Define the model's name for it's output files
        # Load a pre-trained ResNet-34 model and turn off autograd
        # so its weights won't change.
        architecture = resnet34(pretrained=pretrained)
        if freeze_weights:
            for layer in architecture.parameters():
                layer.requires_grad = False
        # Copy the convolutional layers of the model.
        self.conv1 = architecture.conv1
        self.bn1 = architecture.bn1
        self.relu = architecture.relu
        self.maxpool = architecture.maxpool
        self.layer1 = architecture.layer1
        self.layer2 = architecture.layer2
        self.layer3 = architecture.layer3
        self.layer4 = architecture.layer4
        # Copy the average pooling layer of the model.
        self.avgpool = architecture.avgpool
        # Redefine the classification block of ResNet-34.
        # Use LeakyReLU units instead of ReLU units.
        # Output layer has 2 nodes only for the 2 classes in the PCam dataset.
        in_ftrs = architecture.fc.in_features
        self.fc = nn.Linear(in_features=in_ftrs, out_features=2, bias=True)
        # Define a LogSoftmax layer for converting outputs to probabilities
        # Not needed in `forward()` because included in nn.CrossEntropyLoss
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        """Does a forward pass on an image x."""
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)
        out = self.avgpool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out

models.Veggie16 = Veggie16
models.AlmondNet = AlmondNet
models.RaisinNet34 = RaisinNet34

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
        # The directory where the model will be save should exist.
        if not os.path.exists(model_dir) or not os.path.isdir(model_dir):
            raise ValueError(f'Proposed model directory {model_dir} does not exist or is not a folder.')
        self.model_dir = model_dir
        self.checkpoint_file = os.path.join(self.model_dir, model.__class__.__name__ + '_ckpt.pth')
        self.final_file = os.path.join(self.model_dir, model.__class__.__name__ + '_final.pth')

    def train(self, train_loader, criterion, optimizer, num_epochs=25):
        """Trains the model on a training set.

        Args:
            train_loader: A DataLoader to the training data set.
            criterion: Loss function for the model.
            optimizer: Optimization algorithm to be used.
            num_epochs: The number of iterations of the optimizer (default=25).

        Return: (losses, accuracies) over all the epochs.
        """
        self.model.train()
        since = time.time()
        num_steps = len(train_loader)
        losses = []
        accuracies = []
        for epoch in range(1, num_epochs+1):
            epoch_accuracy = 0.0
            epoch_loss = 0.0
            for i, (images, labels) in enumerate(train_loader, start=1):
                images = images.to(self.device)
                labels = labels.to(self.device)
                labels = labels.long().flatten()
                # Forward propagate and evaluate
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                # Backpropagate loss and update weights
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # Compute class probabilities -> predictions -> accuracy
                probabilities = self.model.log_softmax(outputs)
                predictions = torch.argmax(probabilities, dim=1)
                accuracy = (predictions == labels).sum().item() / len(labels)
                # Compute running average of epoch loss
                epoch_loss = (epoch_loss * (i-1) + loss.item()) / i
                # Compute running average of accuracy
                epoch_accuracy = (epoch_accuracy * (i-1) + accuracy) / i
                # Print progress every 1000 batches
                if i % 1000 == 0:
                    print(f'Epoch [{epoch}/{num_epochs}], Step [{i}/{num_steps}], '
                          f'Loss: {loss.item():.6f}; Accuracy: {100*accuracy:.2f}%')
            # Print metrics for the current epoch
            print(f'------------[Loss = {epoch_loss:.6f}; Accuracy = {100*epoch_accuracy:.4f}%]------------')
            # Update trackers
            losses.append(epoch_loss)
            accuracies.append(epoch_accuracy)
            # Save current weights of model in case kernel crashes.
            self.save_checkpoint()
        # Print training time
        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        return losses, accuracies

    def evaluate(self, val_loader: torch.utils.data.DataLoader, criterion):
        """Evaluates the model on a validation set.

        Args:
            model: A PyTorch model.
            val_loader: A DataLoader to the evluation data set.
            device: The CUDA device being used.
            criterion: Loss function for the model.

        Returns: A tuple of the (F1-Score, Accuracy, Total Loss) for the model.
        """
        total_loss = 0.0
        tp, fp, fn, tn = 0, 0, 0, 0
        
        self.model.eval()
        with torch.no_grad():
            since = time.time()
            for i, (images, labels) in enumerate(val_loader):
                images = images.to(self.device)
                labels = labels.to(self.device)
                labels = labels.long().flatten()
                # Forward propagate and evaluate
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                # Compute class probabilities -> predictions
                probabilities = self.model.log_softmax(outputs)
                predictions = torch.argmax(probabilities, dim=1)
                # Compute confusion matrix terms
                tp += (predictions[labels == 1] == 1).sum().item()
                fp += (predictions[labels == 0] == 1).sum().item()
                fn += (predictions[labels == 1] == 0).sum().item()
                tn += (predictions[labels == 0] == 0).sum().item()
                # Compute total loss
                total_loss += loss.item()
        # Compute model accuracy and f1-score
        accuracy = (tp + tn) / (tp + fp + fn + tn + 1e-10)
        score = f1_score(tp, fp, fn, tn)
        return score, accuracy, total_loss
    
    def load_from_file(self, path: str, msg_file_name: str = 'File'):
        """Loads model weights from a file.
        
        Args:
            path: Path to the model weights file.
            msg_file_name: Optional reference for log message.
        """
        if os.path.exists(path) and os.path.isfile(path):
            self.model.load_state_dict(torch.load(path))
            print(f'Successfully Loaded {msg_file_name}: {path}')
        else:
            print(f'No File Found at: {path}')

    def save_to_file(self, path: str, msg_file_name: str = 'Weights'):
        """Saves model weights to a file.
        
        Args:
            path: Path to the model weights file.
            msg_file_name: Optional reference for log message.
        """
        if os.path.exists(os.path.dirname(path)):
            torch.save(self.model.state_dict(), path)
            print(f'Successfully saved {msg_file_name} to: {path}')
        else:
            print(f'Attempted to save {msg_file_name} invalid directory: {path}')

    def save_checkpoint(self):
        """Saves current weights to checkpoint file."""
        self.save_to_file(self.checkpoint_file, 'checkpoint file')

    def load_checkpoint(self):
        """Loads the weights from last checkpoint."""
        self.load_from_file(self.checkpoint_file, 'checkpoint file')
    
    def save_final_model(self):
        """Saves the current weights to the final weights file."""
        self.save_to_file(self.final_file, 'final weights')
        
    def load_final_model(self):
        """Loads model weights from the final weights file."""
        self.load_from_file(self.final_file, 'final weights')

training.Trainer = Trainer


#######################################################
# This module contains code used to evaluate a model. # 
#######################################################
def plot_loss_and_accuracy(losses, accuracies):
    plt.subplots_adjust(wspace=1)
    fig, ax = plt.subplots(1,2, figsize=(10,5))
    ax[0].set_title('Training Loss')
    ax[0].plot(list(range(1,len(losses)+1)), losses, 'r-')
    ax[0].set_ylabel('Categorical Cross Entropy Loss')
    ax[0].set_ylim(0, max(losses)*1.1)
    ax[0].set_xlabel('Epochs')
    ax[1].set_title('Training Accuracy')
    ax[1].plot(list(range(1,len(accuracies)+1)), accuracies, 'b-')
    ax[1].set_ylabel('Accuracy (%)')
    ax[1].set_ylim(min(accuracies)*0.9, max(accuracies)*1.1)
    ax[1].set_xlabel('Epochs')
    return fig, ax

def f1_score(tp: int, fp: int, fn: int, tn: int):
    """Computes the f1_score for a given pair of predictions and ground truths.
    
    Args:
        tp: True positive predictions.
        fp: False positive predictions.
        fn: False negative predictions.
        tn: True negative predictions
        
    Return: F1-Score
    """
    precision = tp / (tp + fp + 1e-10)
    recall = tp / (tp + fn + 1e-10)
    return 2 * precision * recall / (precision + recall + 1e-10)

evaluation.plot_loss_and_accuracy = plot_loss_and_accuracy
evaluation.f1_score = f1_score


#################################################
# Transforms applied to images in this project. #
#################################################
class ToNormalized(object):
    """Perform channel-wise normalization on an RGB image."""
    
    def __init__(self, mean, std):
        """Create a normalizing transform.
        
        Args:
            mean: Channel-wise means for the images. 
            std: Channel-wise stds for the images.
        """
        self.mean = torch.Tensor(mean).reshape(3, 1, 1)
        self.std = torch.Tensor(std).reshape(3, 1, 1)
    
    def __call__(self, pic: torch.Tensor) -> torch.Tensor:
        """
        :param pic: (numpy.ndarray): Image to be normalized

        :return: normalized_image: (numpy.ndarray): Mean-normalized image
        """
        if not isinstance(pic, torch.Tensor):
            raise TypeError(f'Transform {self.__class__.__name__} expects a PyTorch tensor.')
        normalized_image = torch.div(torch.sub(pic, self.mean), self.std)
        return normalized_image

class ToClosed(object):
    """Perform morphological closing on an RGB image with a 2,2 kernel."""

    def __call__(self, img: PIL.Image.Image or np.ndarray) -> PIL.Image.Image:
        if isinstance(img, PIL.Image.Image):
            img = np.array(img)
        elif not isinstance(img, np.ndarray):
            raise TypeError(f'Transform {self.__class__.__name__} expects a PIL image or Numpy array.')
        # Convert to BGR format for compatibility with OpenCV
        cv2_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        closed_img = cv2.morphologyEx(cv2_img, cv2.MORPH_CLOSE, self.kernel)
        # Convert back to RGB format.
        closed_img = cv2.cvtColor(closed_img, cv2.COLOR_BGR2RGB)
        # Convert OpenCV matrix to a PIL Image
        closed_img = PIL.Image.fromarray(np.uint8(closed_img))
        return closed_img

    def __init__(self):
        self.kernel = np.ones([2, 2], np.uint8)

transforms.ToNormalized = ToNormalized
transforms.ToClosed = ToClosed
