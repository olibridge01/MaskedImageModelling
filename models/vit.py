import torch
import torch.nn as nn
from einops.layers.torch import Rearrange

class MLP(nn.Module):
    """Feed-forward neural network for the transformer block."""
    def __init__(self, 
                 dim: int, 
                 inner_dim: int, 
                 dropout: int = 0.0
    ):
        """
        Args:
        - dim (int): Dimension of the input tensor.
        - inner_dim (int): Dimension of the hidden layer.
        - dropout (float): Dropout rate.
        """
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, inner_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        """
        Forward pass.
        
        Args: 
        - x (torch.Tensor): Input tensor.

        Returns:
        - torch.Tensor: MLP output.
        """
        return self.network(x)


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention module."""
    def __init__(self, 
                 dim: int, 
                 heads: int = 8, 
                 dim_head: int = 64, 
                 dropout: float = 0.0
    ):
        """
        Args:
        - dim (int): Dimension of the input tensor.
        - heads (int): Number of attention heads.
        - dim_head (int): Dimension of each attention head.
        - dropout (float): Dropout rate.
        """
        super().__init__()

        # Set multi-head attention parameters
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.scale = dim_head ** -0.5

        # Define layers
        self.dropout = nn.Dropout(dropout)
        self.layernorm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.fc_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        """
        Forward pass of the attention module.
        
        Args:
        - x (torch.Tensor): Input tensor.
        
        Returns:
        - torch.Tensor: Attention output.
        """
        # Apply layer normalization
        x = self.layernorm(x)

        # Get queries, keys, and values
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q = qkv[0].reshape(x.shape[0], x.shape[1], self.heads, -1).transpose(1, 2)
        k = qkv[1].reshape(x.shape[0], x.shape[1], self.heads, -1).transpose(1, 2)
        v = qkv[2].reshape(x.shape[0], x.shape[1], self.heads, -1).transpose(1, 2)

        # Compute attention scores
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        # Apply softmax and dropout
        attention = self.attend(dots)
        attention = self.dropout(attention)

        # Compute attention output
        out = torch.matmul(attention, v)
        out = out.transpose(1, 2).reshape(x.shape[0], x.shape[1], -1)

        # Apply final linear layer
        return self.fc_out(out)


class Transformer(nn.Module):
    """
    Transformer module.
    
    - Originally proposed in 'Attention is All You Need' by Vaswani et al.
    - https://arxiv.org/abs/1706.03762
    """
    def __init__(self, 
                 dim: int, 
                 depth: int, 
                 heads: int, 
                 dim_head: int, 
                 mlp_dim: int, 
                 dropout: float = 0.0
    ):
        """
        Args:
        - dim (int): Dimension of the input tensor.
        - depth (int): Number of transformer blocks.
        - heads (int): Number of attention heads.
        - dim_head (int): Dimension of each attention head.
        - mlp_dim (int): Dimension of the MLP.
        - dropout (float): Dropout rate.
        """
        super().__init__()

        # Layer normalization
        self.norm = nn.LayerNorm(dim)

        # Create transformer blocks
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                MultiHeadAttention(dim=dim, heads=heads, dim_head=dim_head, dropout=dropout),
                MLP(dim=dim, inner_dim=mlp_dim, dropout=dropout)
            ]))

    def forward(self, x):
        """
        Transformer forward pass.
        
        Args:
        - x (torch.Tensor): Input tensor.

        Returns:
        - torch.Tensor: Transformer output.
        """
        for attention, mlp in self.layers:
            x = attention(x) + x
            x = mlp(x) + x

        return self.norm(x)


class ViT(nn.Module):
    """
    Vision Transformer (ViT) module.
    
    - Originally proposed in 'An Image is Worth 16x16 Words: Transformers for 
      Image Recognition at Scale' by Dosovitskiy et al.
    - https://arxiv.org/pdf/2010.11929.pdf

    """
    def __init__(self, 
                 image_size: int, 
                 patch_size: int, 
                 dim: int, 
                 depth: int, 
                 heads: int, 
                 mlp_dim: int, 
                 num_classes: int = 0,
                 n_channels: int = 3, 
                 dim_head: int = 64, 
                 dropout: int = 0.0, 
                 emb_dropout: int = 0.0
    ):
        """
        Args:
        - image_size (int): Size of the image.
        - patch_size (int): Size of the patch.
        - num_classes (int): Number of classes.
        - dim (int): Dimension of the input tensor.
        - depth (int): Number of transformer blocks.
        - heads (int): Number of attention heads.
        - mlp_dim (int): Dimension of the MLP.
        - n_channels (int): Number of channels.
        - dim_head (int): Dimension of each attention head.
        - dropout (float): Dropout rate.
        - emb_dropout (float): Embedding dropout rate.
        """
        super().__init__()

        assert image_size % patch_size == 0, 'Image size must be divisible by patch size!'

        # Get image and patch dimensions
        image_height, image_width = (image_size, image_size)
        patch_height, patch_width = (patch_size, patch_size)

        # Calculate number of patches and patch dimension
        n_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = patch_height * patch_width * n_channels

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h ph) (w pw) -> b (h w) (ph pw c)', ph = patch_height, pw = patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, n_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))

        # Instantiate transformer module for use within the ViT
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        # MLP head for classification tasks - if num_classes is 0, return transformer output
        if num_classes == 0:
            self.head = nn.Identity()
        else:
            self.head = nn.Linear(dim, num_classes)

    def forward(self, img):
        """
        ViT forward pass.
        
        Args:
        - img (torch.Tensor): Input image.

        Returns:
        - torch.Tensor: ViT output.
        """
        # Get patches and patch embeddings
        x = self.to_patch_embedding(img)

        # x.shape = (batch_size, num_patches, dim)
        batch_size, n_patches, _ = x.shape

        # Get class tokens
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)

        # Concatenate class tokens and position embeddings
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n_patches + 1)]

        # Apply dropout and transformer
        x = self.dropout(x)
        x = self.transformer(x)

        # Get class token
        x = x[:, 0]

        # Apply MLP head
        return self.head(x)