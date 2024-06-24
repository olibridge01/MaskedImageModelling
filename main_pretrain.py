# Description: Pre-train SimMIM model on a subset of the ImageNet1k dataset.
import time
from datetime import datetime
import argparse

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from torchvision import datasets
import torchvision.transforms as T

# File imports
from models.vit import ViT
from models.simmim import SimMIM
from utils.utils import get_device, save_reconstructions, pretrain_transforms
from utils.configs import configs

# # Command line arguments
# TRAIN_SIZE = 100000
# BATCH_SIZE = 64
# CONFIG = 'vit_4M_pretrain'
# RUN_PLOTS = False # plot reconstructions every 10 epochs

# Parse arguments from command line
parser = argparse.ArgumentParser(description='Pre-train SimMIM model on a subset of the ImageNet1k dataset.')
parser.add_argument('--config', type=str, default='vit_4M_pretrain', help='Configuration to use for pre-training.')
parser.add_argument('--val_set', action='store_true', help='Whether to use the smaller (validation) set for training.')
parser.add_argument('--train_size', type=int, default=100000, help='Number of training samples to use.')
parser.add_argument('--run_plots', action='store_true', help='Whether to plot reconstructions during training.')

# Get arguments from command line
args = parser.parse_args()
config = configs[args.config]
val_set = args.val_set
train_size = args.train_size
run_plots = args.run_plots

# Device
device = get_device()
print(f'Using device: {device}')

# Dataset transforms
transform = pretrain_transforms(image_size=config['image_size'])

# Load the dataset and sample a random subset
print('Loading dataset (this may take a while)...')

split = 'val' if val_set else 'train'
dataset = datasets.ImageNet(root='./data', split=split, transform=transform)
train_idx = torch.randperm(len(dataset))[:train_size] # random subset indices
train_set = torch.utils.data.Subset(dataset, train_idx)
trainloader = torch.utils.data.DataLoader(train_set, batch_size=config['batch_size'], shuffle=True)

# Load the model
model = ViT(
    image_size = config['image_size'],
    patch_size = config['patch_size'],
    dim = config['dim'],
    depth = config['depth'],
    heads = config['heads'],
    mlp_dim = config['mlp_dim']
).to(device)
    
# Print number of parameters
n_params = sum(p.numel() for p in model.parameters())
print('Number of parameters:', n_params)

# Model save name from parameters
savename = args.config + '_'
savename += 'data_' + str(train_size // 10**3) + 'K'
print(f'Saving model as: {savename}')

mim = SimMIM(
    encoder = model,
    masking_ratio = config['masking_ratio'],
).to(device)
optimizer = optim.AdamW(
        params=mim.parameters(),
        lr=config['lr'],
        weight_decay=config['weight_decay']
)
scheduler = MultiStepLR(optimizer, 
                        milestones=config['schedule_milestones'], 
                        gamma=config['schedule_gamma']
)

for i in range(config['epochs']):
    j = 0
    running_loss = 0.0
    epoch_start = time.time()
    print(f'Epoch {i} - Training...', end='\r')
    for images, _ in trainloader:
        j += 1

        images = images.to(device)
        loss, pred, masks = mim(images)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        running_loss += loss.item()

    # Step the LR scheduler
    scheduler.step()

    print(f'Epoch {i} - Loss: {running_loss / len(trainloader):.5f} - '
        f'Time: {time.time() - epoch_start:.2f}s - '
        f'LR: {scheduler.get_last_lr()[0]:.2e}')

    # Save model (and plot reconstructions)
    if (i + 1) % 10 == 0:
        torch.save(mim.encoder.state_dict(), f'weights/encoder_{savename}.pth')
        torch.save(mim.state_dict(), f'weights/mim_{savename}.pth')
        if run_plots:
            mim.plot_reconstructions(images, savename + f'_epoch_{i + 1}')