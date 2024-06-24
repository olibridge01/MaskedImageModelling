import os
import matplotlib.pyplot as plt

import torch
import torchvision
import torchvision.transforms as T

class AugmentedOxfordIIITPet(torchvision.datasets.OxfordIIITPet):
    """
    Data augmentation class for the Oxford IIIT Pets dataset.
    
    This class is built on the torchvision.datasets.OxfordIIITPet class. It adds transforms such that
    one can resize and transform the images AND the segmentation masks (we want a constant image size
    to pass into the ViT).
    """
    def __init__(
        self,
        root: str,
        split: str,
        target_types: str = "segmentation",
        download: bool = False,
        pre_transform = None,
        post_transform = None,
        pre_target_transform = None,
        post_target_transform = None,
        common_transform = None,
    ):
        """
        Args:
        - root (str): Root directory of dataset.
        - split (str): One of {'test', 'trainval'}.
        - target_types (str): One of {'segmentation', 'category'}.
        - download (bool): Whether to download the dataset.
        - pre_transform (callable): Prior transformation to images.
        - post_transform (callable): Post transformation to images.
        - pre_target_transform (callable): Prior transformation to targets.
        - post_target_transform (callable): Post transformation to targets.
        - common_transform (callable): Common transformation to both images and targets.
        """
        super().__init__(
            root=root,
            split=split,
            target_types=target_types,
            download=download,
            transform=pre_transform,
            target_transform=pre_target_transform,
        )
        self.post_transform = post_transform
        self.post_target_transform = post_target_transform
        self.common_transform = common_transform

    def __len__(self):
        """Return the length of the dataset."""
        return super().__len__()

    def __getitem__(self, idx):
        """Get the item at the given index."""
        (input, target) = super().__getitem__(idx)
        
        # Apply common transform
        if self.common_transform is not None:
            both = torch.cat([input, target], dim=0)
            both = self.common_transform(both)
            (input, target) = torch.split(both, 3, dim=0)
        
        # Apply post transforms to input and target
        if self.post_transform is not None:
            input = self.post_transform(input)
        if self.post_target_transform is not None:
            target = self.post_target_transform(target)

        return (input, target)
    

def get_device():
    """Get the device to be used for training."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")
    
def get_dict(**kwargs):
    """Return a dictionary from keyword arguments."""
    return kwargs

# class ToDevice(torch.nn.Module):
#     """
#     Sends the input object to the device specified in the
#     object's constructor by calling .to(device) on the object.
#     """
#     def __init__(self, device):
#         super().__init__()
#         self.device = device

#     def forward(self, img):
#         """Send the input object to the device."""
#         return img.to(self.device)

#     def __repr__(self) -> str:
#         return f"{self.__class__.__name__}(device={self.device})"


def save_reconstructions(pred_patches: torch.Tensor, 
                         mask_patches: torch.Tensor, 
                         org_patches: torch.Tensor, 
                         model_name: str, 
                         plot_idx: int
):
    """
    Plot and save reconstructions of given image patches.

    Args:
        pred_patches (torch.Tensor): Predicted patches.
        mask_patches (torch.Tensor): Masked patches.
        org_patches (torch.Tensor): Original patches (full image).
        model_name (str): Name of config (for saving the plots at eg. 'figures/vit_4M/').
        plot_idx (int): Index of the image to be plotted (to avoid overwriting).
    """
    device = pred_patches.device
    n_patches = org_patches.size(0)

    # Unnormalize
    mean, std = get_imagenet_defaults()
    mean, std = mean.to(device), std.to(device)
    pred_patches = pred_patches * std + mean
    mask_patches = mask_patches * std + mean
    org_patches = org_patches * std + mean

    # Replace patches with reconstructed patches
    masked_idx = []
    for i in range(n_patches):
        original_patch = org_patches[i] # get ith original patch
        original_patch = original_patch.view(-1) # Flatten

        # Check against all the patches in mask_patches to see if the patch was masked
        for patch_id, mask_patch in enumerate(mask_patches):
            mask_patch = mask_patch.view(-1)
            # If the patch was masked, save the index
            if torch.allclose(original_patch, mask_patch):
                masked_idx.append((i, patch_id)) # (Original patch index, Masked patch index)
                break

    # Replace the masked patches with the reconstructed patches
    reconstruction = org_patches.clone()
    reconstruction[torch.tensor(masked_idx)[:, 0]] = pred_patches[torch.tensor(masked_idx)[:, 1]]

    # Construct original image with masked patches grayed out
    grayed_out = org_patches.clone()
    grayed_out[torch.tensor(masked_idx)[:, 0]] = torch.tensor([0.0, 0.0, 0.0]).to(device)

    savepath = 'figures/' + model_name
    plot_from_patches(grayed_out, n_patches, savepath, f'reconstruction_{int(plot_idx)}_grayed')
    plot_from_patches(reconstruction, n_patches, savepath, f'reconstruction_{int(plot_idx)}')
    plot_from_patches(org_patches, n_patches, savepath, f'reconstruction_org_{int(plot_idx)}')

def plot_from_patches(patches: torch.Tensor, 
                      n_patches: int, 
                      savepath: str, 
                      name: str
):
    """
    Construct image from patches and save it.

    Args:
        patches (torch.Tensor): Patches to be plotted. (n_patches, h, w, c)
        n_patches (int): Number of patches in the image.
        savepath (str): Path to save the image.
        name (str): Name of the image.
    """
    nrow = int(n_patches ** 0.5)
    os.makedirs(savepath, exist_ok=True) # Make directory if it doesn't exist

    patches = torchvision.utils.make_grid(patches.permute(0, 3, 1, 2), nrow=nrow, padding=0)
    plt.imshow(patches.permute(1, 2, 0).detach().cpu())
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(f'{savepath}/{name}.pdf', bbox_inches='tight', pad_inches=0)
    plt.close()

def get_imagenet_defaults():
    """Return the mean and standard deviation of ImageNet."""
    IMAGENET_DEFAULT_MEAN = torch.tensor([0.485, 0.456, 0.406])
    IMAGENET_DEFAULT_STD = torch.tensor([0.229, 0.224, 0.225])

    return IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

def pretrain_transforms(image_size: int):
    """
    Returns the transforms for the pretraining phase.
    
    Args:
    - image_size (int): Size of images to be resized to.
    """
    # Transforms
    IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD = get_imagenet_defaults()

    transform = T.Compose([
        T.Lambda(lambda x: x.convert('RGB') if x.mode != 'RGB' else x),
        T.RandomResizedCrop(
            size=image_size,
            scale=(0.67, 1),
            ratio=(3. / 4., 4. / 3.)
            ),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)
        ])

    return transform

def convert_trimap_seg(map: torch.Tensor) -> torch.Tensor:
    """
    Rescale the segmentation map to 0, 1, 2.
    
    Args:
    - map (torch.Tensor): segmentation map.

    Returns:
    - torch.Tensor: rescaled segmentation map.
    """
    rescaled_map = map * 255
    rescaled_map = rescaled_map.to(torch.long)
    rescaled_map = rescaled_map - 1
    return rescaled_map

def finetune_transforms(image_size: int):
    """
    Returns the transforms for the finetuning phase on Oxford Pet data.
    This ensures transforms are applied to both the images and the targets.

    Args:
    - image_size (int): Size of images to be resized to.
    """
    transform = get_dict(
		pre_transform=T.ToTensor(),
		pre_target_transform=T.ToTensor(),
		common_transform=T.Compose([
			T.Resize((image_size, image_size), interpolation=T.InterpolationMode.NEAREST),
			T.RandomHorizontalFlip(p=0.5),
		]),
		post_transform=T.Compose([
			T.ColorJitter(contrast=0.3)
		]),
		post_target_transform=T.Compose([
			T.Lambda(convert_trimap_seg),
		]),
	)

    return transform

def int_finetune_transforms(image_size: int):
    """
    Returns the transforms for the intermediate fine-tuning phase.
    
    Args:
    - image_size (int): Size of images to be resized to.
    """
    # Transforms
    IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD = get_imagenet_defaults()

    transform = T.Compose([
        T.Lambda(lambda x: x.convert('RGB') if x.mode != 'RGB' else x),
        T.Resize((image_size, image_size), interpolation=T.InterpolationMode.NEAREST),
        T.RandomHorizontalFlip(),
        T.ColorJitter(contrast=0.3),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)
        ])

    return transform