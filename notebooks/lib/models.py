##############################################################
# This module contains the definitions for all convolutional #
# neural networks models used in this project.               #
##############################################################

import torch
import torch.nn as nn
from torchvision.models import vgg16, alexnet


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


class AlmondNet(nn.Module):
    """
    A model that adapts the AlexNet CNN architecture

    This network applies transfer learning to learn the parameters
    of alexnet, and freezes those layers of the model. The classification
    layer of the architecture is modified and will be retrained to
    predict the desired number of output classes.
    """
    
    def __init__(self, num_classes):
        """
        Creates an AlmondNet Network

        :param num_classes: (int): number of different classes to sort the data
         into.
        """
        super(AlmondNet, self).__init__()
        # Load the pretrained AlexNet model, 
        # remove autograd to keep weights from changind
        architecture = alexnet(pretrained=True)
        for layer in architecture.parameters():
            layer.requires_grad = False
        # copy architecture features and avg pool layer
        self.features = architecture.features
        self.avgpool = architecture.avgpool
        in_ftrs = architecture.classifier[1].in_features
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
