import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.utils import save_reconstructions

class SimMIM(nn.Module):
    """
    SimMIM model class.
    
    - Originally proposed in 'SimMIM: a Simple Framework for Masked Image Modeling' by Xie et al.
    - https://arxiv.org/pdf/2111.09886.pdf
    """
    def __init__(
        self,
        encoder: nn.Module,
        masking_ratio: float = 0.5
    ):
        """
        Args:
        - encoder (nn.Module): Vision Transformer to be trained.
        - masking_ratio (float): Ratio of patches to be masked (default 0.5).
        """
        super().__init__()
            
        # Get hyperparameters and functions from encoder (ViT to be trained)
        self.encoder = encoder
        encoder_dim = encoder.pos_embedding.shape[-1]

        assert masking_ratio < 1 and masking_ratio > 0, 'Masking ratio must be between 0 and 1!'
        self.mask_ratio = masking_ratio

        self.get_patches = encoder.to_patch_embedding[0]
        self.get_patch_embedding = nn.Sequential(*encoder.to_patch_embedding[1:])
        patch_values = encoder.to_patch_embedding[2].weight.shape[-1]

        # Linear head (decoder) to predict pixel values
        self.mask_token = nn.Parameter(torch.randn(encoder_dim))
        self.decoder = nn.Linear(encoder_dim, patch_values)

        # L1 Loss function
        self.loss = nn.L1Loss()

    def forward(self, x):
        """
        Run a forward pass of the SimMIM model.

        Args:
        - x (torch.Tensor): Batch of images to be masked and reconstructed.

        Returns:
        - float: Reconstruction loss for the batch.
        - torch.Tensor: Predicted pixel values for masked patches.
        - torch.Tensor: Ground truth masked patches.
        """
        
        # Get device
        device = x.device

        # Get patches, patch embeddings and positional embeddings
        patches = self.get_patches(x)
        batch_size, n_patches, _ = patches.shape
        batch_range = torch.arange(batch_size, device = device)[:, None]

        pos_embedding = self.encoder.pos_embedding[:, 1:(n_patches + 1)]
        tokens = self.get_patch_embedding(patches) + pos_embedding

        # Get mask tokens
        mask_tokens = self.mask_token.unsqueeze(0).expand(batch_size, n_patches, -1) + pos_embedding

        # Calculate number of masked tokens and create a boolean mask
        n_masked = int(self.mask_ratio * n_patches)
        masked_indices = torch.rand(batch_size, n_patches, device=device).topk(k=n_masked, dim=-1).indices
        masked_bool = torch.zeros((batch_size, n_patches), device=device).scatter_(-1, masked_indices, 1).bool()

        # Mask the tokens and run them through the transformer
        tokens = torch.where(masked_bool[:,:, None], mask_tokens, tokens)
        encoded_tokens = self.encoder.transformer(tokens)
        encoded_mask_tokens = encoded_tokens[batch_range, masked_indices]

        # Predict pixel values for masked patches with decoder, and calculate loss (L1)
        pred_patches = self.decoder(encoded_mask_tokens)
        masked_patches = patches[batch_range, masked_indices]
        loss = self.loss(pred_patches, masked_patches) / n_masked

        return loss, pred_patches, masked_patches

    def plot_reconstructions(self, test_images, savename):
        """
        Plot and save reconstructions of random images from the test set.

        Args:
        - testloader (torch.utils.data.DataLoader): Test set loader.
        - savename (str): Name of config (for saving the plots at eg. 'figures/vit_4M/').
        """
        # Get device
        device = test_images.device

        # Evaluate model on test images
        with torch.no_grad():
            test_loss, test_pred, test_masks = self.forward(test_images)

        img_size = test_images.size(-1)
        patch_size = int((test_pred.size(-1) / 3) ** 0.5)

        # Select a random image, get MIM output and ground truth masks
        plot_idx = torch.randint(0, test_images.size(0), (1,)).item()
        pred_patches = test_pred[plot_idx].view(-1, patch_size, patch_size, 3)
        mask_patches = test_masks[plot_idx].view(-1, patch_size, patch_size, 3)

        # Get the full image, split into patches
        org_patches = self.get_patches(test_images[plot_idx].unsqueeze(0)).view(-1, patch_size, patch_size, 3)

        # Plot and save the reconstructions
        save_reconstructions(pred_patches, mask_patches, org_patches, savename, plot_idx)