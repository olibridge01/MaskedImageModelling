# Description: Evaluate the performance and plot results of pre-trained and fine-tuned models
import time
from datetime import datetime
import argparse
import matplotlib.pyplot as plt
import torch
from torchvision import datasets

# File imports
from models.vit import ViT
from models.simmim import SimMIM
from models.finetune import FineTune
from utils.utils import get_device, pretrain_transforms, finetune_transforms, AugmentedOxfordIIITPet
from utils.configs import configs

def pretrain_evaluate(full_mim_path, config, BATCH_SIZE, TEST_SIZE=1000):
    '''
    Save example reconstructions from the pre-trained model.
    '''

    # Device
    device = 'cpu'
    print(f'Using device: {device}')

    # Dataset transforms
    transform = pretrain_transforms(image_size=config['image_size'])

    # Load the validation set and sample a random subset
    print('Loading dataset...')
    dataset = datasets.ImageNet(root='./data', split='val', transform=transform)
    test_idx = torch.randperm(len(dataset))[:TEST_SIZE] # random subset indices
    test_set = torch.utils.data.Subset(dataset, test_idx)
    testloader = torch.utils.data.DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

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

    mim = SimMIM(
        encoder = model,
        masking_ratio = config['masking_ratio'],
    ).to(device)

    #import model weights
    mim.load_state_dict(torch.load(full_mim_path, map_location=device))

    if '/' in full_mim_path: #remove folders from name
        savename = full_mim_path.split('/')[-1].split('.')[0]

    print('Saving reconstructions under:', savename)
    for i in range(20):
        test_images, _ = next(iter(testloader))
        mim.plot_reconstructions(test_images.to(device), savename)

    print('Finished plotting reconstructions')

def finetune_evaluate(config, WEIGHTS_PATH, BATCH_SIZE, TEST_SIZE=1000, TRAIN_SIZE=6000, save=False, show=True):
    '''
    Display/save an example prediction from the segmentation model
    and calculate mIoU, accuracy on the test set.
    '''

    # Seed for reproducibility
    TRAIN_SPLIT_SEED = 42

    # Device
    device = get_device()
    print(f'Using device: {device}')

    # Dataset transforms
    transform = finetune_transforms(image_size=config['image_size'])

    # Download Oxford-IIIT Pet Dataset from PyTorch
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
    splits = [TRAIN_SIZE, TEST_SIZE, len(full_dataset) - TRAIN_SIZE - TEST_SIZE]
    trainset, testset, _ = torch.utils.data.random_split(full_dataset, splits, generator=generator)
    trainset, testset = list(trainset), list(testset) # load the data into RAM (delete if memory runs out)

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

    model = FineTune(
        encoder = encoder,
        weights_path = None
    ).to(device)

    #load model weights
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))

    ### Plot example prediction ###
    #pick image in test set
    idx = torch.randint(0, len(testset), (1,)).item()
    img, target = testset[idx]
    img = img.unsqueeze(0).to(device)
    target = target.unsqueeze(0).to(device)

    img_size = img.shape[-1]
    patch_size = model.patch_size

    with torch.no_grad():
        _, pred_patches, target_patches = model.forward(img, target)

    pred_patches = pred_patches[0]
    target_patches = target_patches[0]

    #make empty tensor to store full image
    pred_full = torch.zeros(1, img_size, img_size)
    target_full = torch.zeros(1, img_size, img_size)

    patch_i = 0
    for row in range(img_size//patch_size):
        for col in range(img_size//patch_size):
            target_plot = target_patches[patch_i].cpu().numpy()
            target_plot = target_plot.reshape(patch_size, patch_size)
            pred_plot = pred_patches[:,patch_i].cpu().numpy()
            pred_plot = pred_plot.reshape(3, patch_size, patch_size)
            
            #take argmax to plot
            pred_plot = pred_plot.argmax(axis=0)

            #add to full image
            target_full[0, row*patch_size:(row+1)*patch_size, col*patch_size:(col+1)*patch_size] = torch.tensor(target_plot)
            pred_full[0, row*patch_size:(row+1)*patch_size, col*patch_size:(col+1)*patch_size] = torch.tensor(pred_plot)
            patch_i += 1
            
    #plot targetand prediction
    fig, axs = plt.subplots(1, 3, figsize=(6, 2))
    axs[0].imshow(img[0].cpu().numpy().transpose(1, 2, 0))
    axs[0].set_title('Image')
    axs[1].imshow(target_full[0].cpu().numpy(), cmap='gray')
    axs[1].set_title('Target')
    axs[2].imshow(pred_full[0].cpu().numpy(), cmap='gray')
    axs[2].set_title('Prediction')


    for ax in axs:
        ax.axis('off')
    if save:
        plt.savefig('figures/finetune_example.png')
    if show:
        plt.show()
    plt.close()

    ### Calculate mIoU and accuracy ###
    #iterate through test set
    test_mIoU = 0
    test_accuracy = 0
    with torch.no_grad():
        for img, target in testset:
            img = img.unsqueeze(0)
            target = target.unsqueeze(0)
            img = img.to(device)
            target = target.to(device)

            loss, pred_patches, target_patches = model.forward(img, target)
            pred_patches = pred_patches[0]
            target_patches = target_patches[0]

            pred_flat = pred_patches.argmax(dim=0).view(-1).cpu()
            target_flat = target_patches.view(-1).cpu()

            #calculate accuracy
            test_accuracy += (pred_flat == target_flat).sum().item()/len(pred_flat)

            #calculate mIoU
            iou = 0
            for j in range(3):
                intersection = ((pred_flat == j) & (target_flat == j)).sum().item()
                union = ((pred_flat == j) | (target_flat == j)).sum().item()
                if union != 0:
                    iou += intersection/union

            test_mIoU += iou/3

    test_mIoU /= len(testset)
    test_accuracy /= len(testset)

    print(f'Test mIoU: {test_mIoU:.4f}')
    print(f'Test accuracy: {test_accuracy:.4f}')

    #iterate through train set
    train_mIoU = 0
    train_accuracy = 0
    with torch.no_grad():
        for img, target in trainset:
            img = img.unsqueeze(0)
            target = target.unsqueeze(0)
            img = img.to(device)
            target = target.to(device)

            loss, pred_patches, target_patches = model.forward(img, target)
            pred_patches = pred_patches[0]
            target_patches = target_patches[0]

            pred_flat = pred_patches.argmax(dim=0).view(-1).cpu()
            target_flat = target_patches.view(-1).cpu()

            #calculate accuracy
            train_accuracy += (pred_flat == target_flat).sum().item()/len(pred_flat)

            #calculate mIoU
            iou = 0
            for j in range(3):
                intersection = ((pred_flat == j) & (target_flat == j)).sum().item()
                union = ((pred_flat == j) | (target_flat == j)).sum().item()
                if union != 0:
                    iou += intersection/union

            train_mIoU += iou/3

    train_mIoU /= len(trainset)
    train_accuracy /= len(trainset)

    print(f'Train mIoU: {train_mIoU:.4f}')
    print(f'Train accuracy: {train_accuracy:.4f}')

if __name__ == '__main__':

    # Parse arguments
    parser = argparse.ArgumentParser(description='Evaluate the performance of pre-trained and fine-tuned models.')
    parser.add_argument('--model', type=str, default='ft', help='Model to evaluate (pt: pre-trained, ft: fine-tuned).')
    parser.add_argument('--config', type=str, default='vit_4M_finetune', help='Configuration to use for pre-training or fine-tuning.')
    parser.add_argument('--train_size', type=int, default=6000, help='Number of fine-tuning training samples to use.')
    parser.add_argument('--test_size', type=int, default=1000, help='Number of fine-tuning test samples to use.')
    parser.add_argument('--weights', type=str, default='weights/vit_4M_finetune', help='Path to pre-trained (MIM) or fine-tuned weights.')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for evaluation.')

    # Get arguments from command line
    args = parser.parse_args()

    config = configs[args.config]
    train_size = args.train_size
    test_size = args.test_size
    weights_path = args.weights
    BATCH_SIZE = args.batch_size

    if args.model == 'pt':
        pretrain_evaluate(weights_path, config, BATCH_SIZE, TEST_SIZE=test_size)

    if args.model == 'ft':
        finetune_evaluate(config, weights_path, BATCH_SIZE, TEST_SIZE=test_size, TRAIN_SIZE=train_size)