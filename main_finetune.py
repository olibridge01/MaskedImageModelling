# Description: Fine-tune a pre-trained ViT model on the Oxford-IIIT Pet dataset for segmentation.
import time
from datetime import datetime
import matplotlib.pyplot as plt
import zipfile
import os
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
import torch.utils
import torchvision.transforms as T
import torchvision
from torchvision.transforms.v2 import MixUp

# File imports
from models.vit import ViT
from models.finetune import FineTune
from utils.utils import AugmentedOxfordIIITPet, get_device, finetune_transforms, int_finetune_transforms
from utils.configs import configs

# Parse arguments
parser = argparse.ArgumentParser(description='Fine-tune a pre-trained ViT model on the Oxford-IIIT Pet dataset for segmentation.')
parser.add_argument('--config', type=str, default='vit_4M_finetune', help='Configuration to use for fine-tuning.')
parser.add_argument('--train_size', type=int, default=6000, help='Number of fine-tuning training samples to use.')
parser.add_argument('--test_size', type=int, default=1000, help='Number of fine-tuning test samples to use.')
parser.add_argument('--weights', type=str, default=None, help='Path to pre-trained weights. If None, the model is trained from scratch.')
parser.add_argument('--int_finetune', action='store_true', help='Whether to perform intermediate fine-tuning on the classification task.')

# Get arguments from command line
args = parser.parse_args()
if args.int_finetune:
	assert args.weights is not None, 'Intermediate fine-tuning requires pre-trained weights.'

config = configs[args.config]
train_size = args.train_size
test_size = args.test_size
weights_path = args.weights
int_finetune = args.int_finetune
NUM_CLASSES = 6 # Number of classes in the intermediate classification dataset
TRAIN_SPLIT_SEED = 42 # Seed for splitting the dataset

# Device
device = get_device()
print(f'Using device: {device}')

# Dataset transforms
transform = finetune_transforms(config['image_size'])

# Download Oxford-IIIT Pet Dataset from PyTorch
print('Loading segmentation dataset...')
trainset = AugmentedOxfordIIITPet(
	root='data',
	split="trainval",
	target_types="segmentation",
	download=True,
	**transform,
)
testset = AugmentedOxfordIIITPet(
	root='data',
	split="test",
	target_types="segmentation",
	download=True,
	**transform,
)
generator = torch.Generator().manual_seed(TRAIN_SPLIT_SEED)
full_dataset = torch.utils.data.ConcatDataset([trainset, testset])

# Resplit full dataset into train and test sets of desired sizes
splits = [train_size, test_size, len(full_dataset) - train_size - test_size]
trainset, testset, _ = torch.utils.data.random_split(full_dataset, splits, generator=generator)
trainset, testset = list(trainset), list(testset) # load the data into RAM

trainloader = torch.utils.data.DataLoader(trainset, batch_size=config['batch_size'], shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=config['batch_size'], shuffle=True)

encoder = ViT(
	image_size = config['image_size'],
	patch_size = config['patch_size'],
	dim = config['dim'],
	depth = config['depth'],
	heads = config['heads'],
	mlp_dim = config['mlp_dim'],
).to(device)

# Print number of parameters
n_params_enc = sum(p.numel() for p in encoder.parameters())
print('Number of parameters (encoder):', n_params_enc)

# Model save name from parameters
savename = args.config + '_'
savename += 'data_' + str(train_size)
# savename += datetime.now().strftime('%H-%M') # to avoid overwriting
print(f'Saving model as: {savename}')

# Run intermediate fine-tuning if specified in command line
if int_finetune:

	# Extract intermediate dataset
	print('Loading intermediate dataset...')
	zip_path = 'data/archive.zip'
	with zipfile.ZipFile(zip_path, 'r') as zip_ref:
		zip_ref.extractall('data')

	# Load intermediate train and test sets
	int_train_path = 'data/seg_train/seg_train'
	int_test_path = 'data/seg_test/seg_test'
	int_transform = int_finetune_transforms(config['image_size'])

	int_trainset = torchvision.datasets.ImageFolder(int_train_path, transform=int_transform)
	int_testset = torchvision.datasets.ImageFolder(int_test_path, transform=int_transform)

	# Get dataloaders
	int_trainloader = torch.utils.data.DataLoader(int_trainset, batch_size=config['batch_size'], shuffle=True)
	int_testloader = torch.utils.data.DataLoader(int_testset, batch_size=config['batch_size'], shuffle=False)

	# Load pre-trained encoder weights
	encoder.load_state_dict(torch.load(weights_path, map_location=device))

	# Add classification head
	encoder.head = nn.Linear(config['dim'], NUM_CLASSES).to(device)

	# Define loss function, optimizer, MixUp and scheduler
	int_criterion = nn.CrossEntropyLoss()
	int_optimizer = optim.Adam(encoder.parameters(), lr=config['int_lr'])
	mixup = MixUp(alpha=config['mixup_alpha'], num_classes=NUM_CLASSES)
	int_scheduler = MultiStepLR(int_optimizer, milestones=config['int_schedule_milestones'], gamma=config['int_schedule_gamma'])
	print('=' * 100)
	print('Running intermediate fine-tuning...')

	# Run intermediate fine-tuning on classification task
	for i in range(config['int_epochs']):
		epoch_start = time.time()
		print(f'Epoch {i} - Training...', end='\r')
		running_loss = 0.0
		encoder.train()
		for i, data in enumerate(int_trainloader):
			inputs, labels = data
			inputs, labels = mixup(inputs, labels)
			inputs, labels = inputs.to(device), labels.to(device)

			int_optimizer.zero_grad()
			outputs = encoder(inputs)

			loss = int_criterion(outputs, labels)
			loss.backward()
			int_optimizer.step()

			running_loss += loss.item()

		int_scheduler.step()
		print(f'Epoch {i} - Loss: {running_loss / len(int_trainloader):.5f} - '
        	f'Time: {time.time() - epoch_start:.2f}s - '
        	f'LR: {int_scheduler.get_last_lr()[0]:.2e}')

	# Save the intermediate fine-tuned encoder weights
	torch.save(encoder.state_dict(), f'weights/int_{savename}.pth')

print('=' * 100)
if args.weights is not None:
	print('Running segmentation fine-tuning...')
else:
	print('Running baseline segmentation training...')
# Segmentation fine-tuning
if int_finetune:
	model = FineTune(
		encoder = encoder,
		weights_path = None,
		weights_device = device
	).to(device)
else:
	model = FineTune(
		encoder = encoder,
		weights_path = weights_path,
		weights_device = device,
	).to(device)
optimizer = optim.AdamW(
        params=model.parameters(),
        lr=config['lr'],
        weight_decay=config['weight_decay']
)
scheduler = MultiStepLR(optimizer, 
                        milestones=config['schedule_milestones'], 
                        gamma=config['schedule_gamma']
)
# Fine-tune the model on segmentation task
for i in range(config['epochs']):
	j = 0
	running_loss = 0.0
	epoch_start = time.time()
	print(f'Epoch {i} - Training...', end='\r')
	model.train()
	for inputs, labels in trainloader:

		inputs, labels = inputs.to(device), labels.to(device)

		optimizer.zero_grad()
		loss, _, _ = model(inputs, labels)
		loss.backward()
		optimizer.step()

		running_loss += loss.item()
		j += 1

	# Step the LR scheduler
	scheduler.step()

	print(f'Epoch {i} - Loss: {running_loss / len(trainloader):.5f} - '
        f'Time: {time.time() - epoch_start:.2f}s - '
        f'LR: {scheduler.get_last_lr()[0]:.2e}')

	# Save model every 10 epochs
	if (i + 1) % 10 == 0:
		torch.save(model.state_dict(), f'weights/{savename}.pth')