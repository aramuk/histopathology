# This module contains code used to train a model.

import time

import torch

def train(model, train_loader, device, criterion, optimizer, num_epochs=25):
	"""Trains a given model.
	
	Args:
		model: a PyTorch model.
		train_loader: a DataLoader to the training data set.
		criterion: Loss function for the model.
		optimizer: Optimization algorithm to be used.
		num_epochs: The number of iterations of the optimizer.

	Return: torch.FloatTensor of loss over epochs.
	"""
	model.train()
	since = time.time()
	num_steps = len(train_loader)
	losses = []
	for epoch in range(1, num_epochs+1):
		epoch_loss = 0.0
		for i, (images, labels) in enumerate(train_loader, start=1):
			images = images.to(device)
			labels = labels.to(device)
			# Generate prediciton and evaluate
			outputs = model(images)
			loss = criterion(outputs, labels.flatten())
			# Backpropagate loss and update weights
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			# Compute running average of epoch loss
			epoch_loss = (epoch_loss * (i-1) + loss.item()) / i
			# Print progress every 1000 batches
			if i % 1000 == 0:
				print(f'Epoch [{epoch}/{num_epochs}], Step [{i}/{num_steps}], Loss: {loss.item():.6f}')
    # Print training time
	time_elapsed = time.time() - since
	print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
	return torch.FloatTensor(loss)

	
