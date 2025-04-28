import torch
import torch.nn as nn
import torch.nn.functional as F


class InputProjection(nn.Module):
    def __init__(self, image_size=28, patch_size=14, model_dim=64):
        super(InputProjection, self).__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.model_dim = model_dim

        # Calculate number of patches
        self.num_patches = (image_size // patch_size) ** 2

        self.projection = nn.Linear(patch_size * patch_size, model_dim)

    def forward(self, x):
        # x shape: (batch_size, num_channels, height, width)
        batch_size = x.size(0)
        # Unfold the image into patches: unfold over the height and width dimension (2 and 3)
        patches = x.unfold(2, self.patch_size, self.patch_size).unfold(
            3, self.patch_size, self.patch_size
        )
        # Reshape patches to (batch_size, num_patches, patch_size * patch_size) so [batch_size, 4, 196]
        patches = patches.contiguous().view(
            batch_size, self.num_patches, self.patch_size * self.patch_size
        )
        # Apply linear projection to each patch
        # [batch_size, num_patches, patch_size * patch_size] -> [batch_size, num_patches, model_dim]
        patches = self.projection(patches)
        return patches


def scaled_dot_product_attention(query, key, value):
    """
    Compute the scaled dot-product attention.
    Args:
        query: Queries of shape (batch_size, num_heads, num_patches, head_dim)
        key: Keys of shape (batch_size, num_heads, num_patches, head_dim)
        value: Values of shape (batch_size, num_heads, num_patches, head_dim)
    Returns:
        output: Attention output of shape (batch_size, num_heads, num_patches, head_dim)
        attention: Attention weights of shape (batch_size, num_heads, num_patches, num_patches)
    """
    d_k = query.size(-1)  # Dimension of keys
    scores = torch.matmul(query, key.transpose(-2, -1)) / (
        d_k**0.5
    )  # (batch_size, num_heads, num_patches, num_patches)
    attention = F.softmax(
        scores, dim=-1
    )  # (batch_size, num_heads, num_patches, num_patches)
    output = torch.matmul(attention, value)  # Weighted sum of values
    return output, attention


class MultiHeadAttention(nn.Module):
    def __init__(self, model_dim=64, num_heads=8):
        super(MultiHeadAttention, self).__init__()
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads

        # Linear layers for Q, K, V
        self.query_fc = nn.Linear(model_dim, model_dim)
        self.key_fc = nn.Linear(model_dim, model_dim)
        self.value_fc = nn.Linear(model_dim, model_dim)

        # Output linear layer
        self.fc_out = nn.Linear(model_dim, model_dim)

    def forward(self, x):
        # x shape: (batch_size, num_patches, model_dim)
        batch_size = x.size(0)

        # Linear projections
        query = self.query_fc(x)
        key = self.key_fc(x)
        value = self.value_fc(x)
        # Split into multiple heads
        query = query.view(batch_size, -1, self.num_heads, self.head_dim).transpose(
            1, 2
        )  # [batch_size, num_heads, num_patches, head_dim]
        key = key.view(batch_size, -1, self.num_heads, self.head_dim).transpose(
            1, 2
        )  # [batch_size, num_heads, num_patches, head_dim]
        value = value.view(batch_size, -1, self.num_heads, self.head_dim).transpose(
            1, 2
        )  # [batch_size, num_heads, num_patches, head_dim]

        # Compute attention per head
        output, _ = scaled_dot_product_attention(query, key, value)

        # Concatenate heads
        output = (
            output.transpose(1, 2).contiguous().view(batch_size, -1, self.model_dim)
        )

        # Final linear layer
        # (batch_size, num_patches, num_heads * head_dim) -> (batch_size, num_patches, model_dim)
        output = self.fc_out(output)
        return output


class FeedForwardNetwork(nn.Module):
    def __init__(self, model_dim=64, ff_dim=512):
        super(FeedForwardNetwork, self).__init__()
        self.fc1 = nn.Linear(model_dim, ff_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(ff_dim, model_dim)

    def forward(self, x):
        # x shape: (batch_size, num_patches, model_dim)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, model_dim=64, num_heads=8, ff_dim=512):
        super(EncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(model_dim, num_heads)
        self.ffn = FeedForwardNetwork(model_dim, ff_dim)
        self.layernorm1 = nn.LayerNorm(model_dim)  # LayerNorm for attention output
        self.layernorm2 = nn.LayerNorm(model_dim)  # LayerNorm for FFN output

    def forward(self, x):
        # x shape: (batch_size, num_patches, model_dim)

        # Attention
        attention_output = self.attention(x)
        x = self.layernorm1(x + attention_output)  # Residual connection

        # Feed Forward Network
        ffn_output = self.ffn(x)
        x = self.layernorm2(x + ffn_output)
        return x


class TransformerEncoder(nn.Module):
    def __init__(
        self, model_dim=64, num_heads=8, ff_dim=512, num_layers=4, max_length=4
    ):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList(
            [EncoderLayer(model_dim, num_heads, ff_dim) for _ in range(num_layers)]
        )
        self.positional_encoding = nn.Parameter(
            torch.randn(1, max_length, model_dim)
        )  # max_length is the number of patches

    def forward(self, x):
        # x shape: (batch_size, num_patches, model_dim)
        x = x + self.positional_encoding
        for layer in self.layers:
            x = layer(x)
        return x


class TransformerMNIST(nn.Module):
    def __init__(
        self,
        image_size=28,
        patch_size=14,
        model_dim=64,
        num_heads=8,
        ff_dim=512,
        num_layers=4,
    ):
        super(TransformerMNIST, self).__init__()
        self.input_projection = InputProjection(image_size, patch_size, model_dim)
        self.encoder = TransformerEncoder(model_dim, num_heads, ff_dim, num_layers)
        self.fc = nn.Linear(model_dim, 10) # classifier

    def forward(self, x):
        # x shape: (batch_size, num_channels, height, width)
        x = self.input_projection(x)  # [batch_size, num_patches, model_dim]
        x = self.encoder(x)
        x = x.mean(dim=1)  # Global average pooling
        x = self.fc(x)  # [batch_size, 10]
        return x
