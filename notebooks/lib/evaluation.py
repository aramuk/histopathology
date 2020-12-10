#######################################################
# This module contains code used to evaluate a model. # 
#######################################################

import time

from sklearn.metrics import confusion_matrix
import torch


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
            loss = criterion(outputs, labels.long().flatten())
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
