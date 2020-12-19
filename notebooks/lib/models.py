##############################################################
# This module contains the definitions for all convolutional #
# neural networks models used in this project.               #
##############################################################

import torch
import torch.nn as nn

from torchvision.models import alexnet, resnet34, vgg16

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