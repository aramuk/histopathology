####################################################
# This module contains code used to train a model. #
####################################################

import os
import time

import torch


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
