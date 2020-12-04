# This module contains code used to train and evaluate a model 

import time
import torch

def train(model, train_loader, device, criterion, optimizer, num_epochs=25):
	"""Trains a given model.
	
	Args:
		model - a PyTorch model.
		train_loader - a DataLoader to the training data set.
		criterion - loss function for the model.
		optimizer - optimization algorithm to user while training.
		num_epochs - the number of iterations to train.
	"""
	model.train()
	since = time.time()
	num_steps = len(train_loader)
	for epoch in range(1, num_epochs+1):
		for i, (images, labels) in enumerate(train_loader, start=1):
			images = images.to(device)
			labels = labels.to(device)

			outputs = model(images)
			loss = criterion(outputs, labels)

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			if i % 10 == 0:
				print(f'Epoch [{epoch}/{num_epochs}], Step [{i}/{num_steps}], Loss: {loss.item():.6f}')

def evaluate(model, test_loader, device):
	"""Evaluates a given model.
	
	Args:
		model - a PyTorch model.
		test_loader - a DataLoader to the evluation data set.
	"""
	model.eval()
	with torch.no_grad():
		correct = 0
		total = 0
		for images, labels in test_loader:
			images = images.to(device)
			labels = labels.to(device)
        
			outputs = model(images)
			predictions = torch.argmax(outputs, dim=1)

			total += labels.size(0)
			correct += (predictions == labels).sum().item()
		print(f'Test Accuracy of Veggie16: {(100*(correct/total)):.6f}')

def train_xval(model, train_loader, val_loader):
	"""Train a model with cross validation."""
	pass
