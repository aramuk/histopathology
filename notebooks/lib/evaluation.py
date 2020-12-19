#######################################################
# This module contains code used to evaluate a model. # 
#######################################################

import csv
import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve
import torch

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

def predict_to_csv(model, unlabeled_loader, device, col_names, csv_path, batch_size=50):
    """Saves model predictions to a CSV file.
    
    Args:
        model: A PyTorch model.
        unlabeled_loader: A PyTorch DataLoader that returns only images.
        device: Device for running model.
        col_names: Column names for the output csv.
        csv_path: Output path of csv.
    """
    if not os.path.exists(os.path.dirname(csv_path)):
        raise ValueError(f'Attempted to save predictions invalid directory: {csv_path}')
    model.eval()
    with torch.no_grad():
        with open(csv_path, 'w') as csvfile:
            predictions_writer = csv.writer(csvfile)
            predictions_writer.writerow(['id','label'])

            num_steps = len(unlabeled_loader)
            for i, images in enumerate(unlabeled_loader):
                images = images.to(device)
                outputs = model(images)
                probabilities = model.log_softmax(outputs)
                predictions = torch.argmax(probabilities, dim=1)

                for j, prob in enumerate(predictions):
                    idx = i*batch_size + j
                    predictions_writer.writerow([os.path.splitext(col_names[idx])[0], 
                                                 prob.item()])
                if i % 100 == 0:
                    print(f'Predictions written for batch [{i}/{num_steps}]')
    print(f'Saved model predictions to: {csv_path}')

def roc(model, data_loader, device, subsample=1.0):
    """Estimate the ROC curve and its integral for the model on a dataset.
    
    Args:
        model: A PyTorch model.
        data_loader: A PyTorch DataLoader (shuffled).
        device: Device for running model.
        subsample: The number of samples to use when estimating the ROC curve.

    Return:
        auc: Area under the ROC curve.
        fpr: False positive rate of model at various thresholds.
        tpr: True positive rate of model at same thresholds as FPR.
        thresholds: Thresholds at which the model was decided
    """
    y_true = []
    y_hat = []
    with torch.no_grad():
        num_batches = len(data_loader)
        sample_size = int(subsample * num_batches)
        
        for i, (images, labels) in enumerate(data_loader, start=1):
            if i > sample_size:
                break
            images = images.to(device)
            labels = labels.long().flatten().to(device)
            # Forward pass and get predicted label
            outputs = model(images)
            probabilities = model.log_softmax(outputs)
            # Update
            y_hat.extend(probabilities[:,1].tolist())
            y_true.extend(labels.tolist())
            if i % 100 == 0:
                print(f'Computed predictions for sample [{i}/{sample_size}]')
    fpr, tpr, thresholds = roc_curve(y_true, y_hat)
    auc = np.trapz(tpr, fpr)
    return auc, fpr, tpr, thresholds

def plot_roc(fpr, tpr, auc, model_name):
    """Plot ROC curve for a model.
    
    Args:
        fpr: False positive rate at various thresholds.
        tpr: True positive rate at same thresholds as fpr.
        auc: Computer area under the ROC curve.
        model_name: The name of the model.
    """
    plt.figure()
    # ROC Curve
    plt.plot(fpr, tpr, 'b-', lw=2, label=f'ROC curve (Area = {auc:.4f})')
    # No discrimination line
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.plot(fpr, tpr)
    # Scale axes
    plt.xlim([-0.05, 1.0])
    plt.ylim([0.0, 1.05])
    # Labels
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.title(f'Received Operating Characteristic Curve for {model_name}')
