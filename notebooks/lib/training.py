####################################################
# This module contains code used to train a model. #
####################################################

import os
import time

import torch

from evaluation import f1_score

class Trainer:
    """A wrapper that trains a model """

    def __init__(self, model, device, train_loader, val_loader, model_dir='/kaggle/working/'):
        """Initialize a Trainer.

        Args:
            model: A PyTorch model
            device: Device to train the model on.
            model_dir: Where to save the model's checkpoint files (default=/kaggle/working/)
        """
        self.model = model
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        # The directory where the model will be save should exist.
        if not os.path.exists(model_dir) or not os.path.isdir(model_dir):
            raise ValueError(f'Proposed model directory {model_dir} does not exist or is not a folder.')
        self.model_dir = model_dir
        self.checkpoint_file = os.path.join(self.model_dir, model.__class__.__name__ + '_ckpt.pth')
        self.final_file = os.path.join(self.model_dir, model.__class__.__name__ + '_final.pth')
        self.epoch = 1
        self.total_epochs = 0

    def reset_epochs(self):
        self.epoch = 1
        self.total_epochs = 0

    def _train_one_epoch(self, criterion, optimizer, scheduler=None, output_freq=1000):
        """Trains the model for one epoch on the training set.

        Args:
            criterion: Loss function for the model.
            optimizer: Optimization algorithm to be used.
            output_freq: Print training metrics every `output_freq` steps.

        Returns: (epoch_loss, epoch_accuracy)
        """
        self.model.train()
        num_steps = len(self.train_loader)
        epoch_accuracy = 0.0
        epoch_loss = 0.0
        for i, (images, labels) in enumerate(self.train_loader, start=1):
            images = images.to(self.device)
            labels = labels.to(self.device).long().flatten()
            # Clear previous gradients
            optimizer.zero_grad()
            # Forward propagate and evaluate
            outputs = self.model(images)
            loss = criterion(outputs, labels)
            # Backpropagate loss and update weights
            loss.backward()
            optimizer.step()
            # Compute class probabilities -> predictions -> accuracy
            probabilities = self.model.log_softmax(outputs)
            predictions = torch.argmax(probabilities, dim=1)
            accuracy = (predictions == labels).sum().item() / len(labels)
            # Compute total epoch loss
            epoch_loss += loss.item()
            # Compute running average of accuracy
            epoch_accuracy = (epoch_accuracy * (i-1) + accuracy) / i
            # Update scheduler if provided:
            if scheduler:
                scheduler.step()
            # Print progress every 1000 batches
            if i % output_freq == 0:
                print(f'Epoch [{self.epoch}/{self.total_epochs}], Step [{i}/{num_steps}], '
                      f'Loss: {loss.item():.6f}; Accuracy: {100*accuracy:.2f}%')
        # Print metrics for the current epoch
        print(f'------------[Loss = {epoch_loss:.6f}; Accuracy = {100*epoch_accuracy:.4f}%]------------')
        # Increment epoch
        self.epoch += 1
        # Return trackers
        return epoch_loss, epoch_accuracy

    def train(self, criterion, optimizer, scheduler=None, num_epochs=25, output_freq=1000):
        """Trains the model on a training set.

        Args:
            train_loader: A DataLoader to the training data set.
            criterion: Loss function for the model.
            optimizer: Optimization algorithm to be used.
            num_epochs: The number of iterations of the optimizer (default=25).

        Return: (train_losses, val_losses, train_accuracies, val_accuracies) over all the epochs.
        """
        train_losses = []
        val_losses = []
        train_accuracies = []
        val_accuracies = []
        self.model.train()
        self.total_epochs += num_epochs
        since = time.time()
        for epoch in range(self.epoch, self.total_epochs+1):
            # Train one epoch
            epoch_loss, epoch_accuracy = self._train_one_epoch(criterion, optimizer, 
                                                               scheduler=scheduler, 
                                                               output_freq=output_freq)
            # Evaluate model
            _, val_accuracy, val_loss = self.evaluate(criterion)
            # Update trackers
            train_losses.append(epoch_loss)
            train_accuracies.append(epoch_accuracy)
            val_losses.append(val_loss)
            val_accuracies.append(val_accuracy)
            # Save current weights of model in case kernel crashes.
            self.save_checkpoint()
            # Update learning rate scheduler
            if scheduler:
                scheduler.step()
        # Print training time
        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        return train_losses, val_losses, train_accuracies, val_accuracies

    def evaluate(self, criterion):
        """Evaluates the model on a validation set.

        Args:
            model: A PyTorch model.
            val_loader: A DataLoader to the evluation data set.
            device: The CUDA device being used.
            criterion: Loss function for the model.

        Returns: A tuple of the (F1-Score, Accuracy, Total Loss) on the validation set.
        """
        total_loss = 0.0
        tp, fp, fn, tn = 0, 0, 0, 0
        self.model.eval()
        with torch.no_grad():
            since = time.time()
            for i, (images, labels) in enumerate(self.val_loader):
                images = images.to(self.device)
                labels = labels.to(self.device).long().flatten()
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
