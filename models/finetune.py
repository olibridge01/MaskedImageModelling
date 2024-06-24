# Description: FineTune model class for training a Vision Transformer on a segmentation task.

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
import matplotlib.pyplot as plt


class FineTune(nn.Module):
    """
    FineTune module.
    
    This model takes as an argument a ViT encoder pre-trained with SimMIM. 
    The model is then fine-tuned on an image segmentation task.
    """
    def __init__(
        self,
        *,
        encoder: nn.Module,
        weights_path: str = None,
        weights_device: str = 'cpu'
    ):
        """
        Args:
        - encoder (nn.Module): Vision Transformer to be trained.
        - weights_path (str): Path to saved weights from pre-trained model. If None, the model is 
        - trained from scratch (baseline model).
        - weights_device (str): Device that the loaded weights are stored on.
        """
        super().__init__()

        # Instantiate encoder (ViT to be fine-tuned)
        self.encoder = encoder
        encoder_dim = encoder.pos_embedding.shape[-1]

        # Load weights from pre-trained encoder
        if weights_path is not None:
            self.encoder.load_state_dict(torch.load(weights_path, map_location=weights_device))

        # Get patches and patch embeddings
        self.get_patches = encoder.to_patch_embedding[0]
        self.get_patch_embedding = nn.Sequential(*encoder.to_patch_embedding[1:])
        patch_values = encoder.to_patch_embedding[2].weight.shape[-1]

        # Infer patch size from above 
        self.patch_size = int((patch_values / 3) ** 0.5)

        # Linear head (decoder) to predict segmentation target
        self.mlp = nn.Linear(encoder_dim, self.patch_size ** 2 * 3)
        self.loss = nn.CrossEntropyLoss()

    def forward(self,
                img: torch.Tensor,
                target: torch.Tensor
    ) -> tuple:
        """
        Run a forward pass of the FineTune model.

        Args:
            img (torch.Tensor): Input image tensor.
            target (torch.Tensor): Target segmentation tensor.

        Returns:
            loss (torch.Tensor): CrossEntropy loss.
            pred_patches (torch.Tensor,
                shape=(batch_size, 3, num_patches, patch_size, patch_size)):
                Predicted pixel values (one hot encoded) organised in patches.
            target_patches (torch.Tensor,
                shape=(batch_size, 3, num_patches, patch_size, patch_size)):
                Target pixel values (one hot encoded) organised in patches.
        """
        # Get device ('cuda' or 'cpu')
        device = img.device

        # Get patches
        patches = self.get_patches(img)
        batch_size, n_patches, _ = patches.shape

        # Get position embeddings
        pos_embedding = self.encoder.pos_embedding[:, 1:(n_patches + 1)]

        # Get encoder output
        tokens = self.get_patch_embedding(patches) + pos_embedding
        encoder_output = self.encoder.transformer(tokens)
        
        # Pass each patch through the prediction head (decoder)
        mlp_output = self.mlp(encoder_output)
        mlp_output = mlp_output.reshape(batch_size, n_patches, 3, -1)
        mlp_output = mlp_output.permute(0, 2, 1, 3).reshape(batch_size, 3, -1)
        mlp_output = nn.functional.log_softmax(mlp_output, dim=-1)

        pred_patches = mlp_output.reshape(batch_size, 3, n_patches, self.patch_size, self.patch_size)
        
        # Convert target to patches and one hot encode
        target_patches = self.get_patches(target)
        target_flat = target_patches.reshape(batch_size, -1)

        # Calculate loss
        loss = self.loss(mlp_output, target_flat)

        return loss, pred_patches, target_patches
    
    def display_example(self, testset, show=False, save=True):
        '''Display/save an example prediction from the segmentation model'''

        #pick image in test set
        idx = torch.randint(0, len(testset), (1,)).item()
        img, target = testset[idx]
        img = img.unsqueeze(0)
        target = target.unsqueeze(0)

        img_size = img.shape[-1]
        patch_size = self.patch_size

        with torch.no_grad():
            _, pred_patches, target_patches = self.forward(img, target)

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
            plt.savefig('finetune_example.png')
        if show:
            plt.show()
        plt.close()