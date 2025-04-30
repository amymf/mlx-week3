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


def scaled_dot_product_attention(query, key, value, use_mask=False):
    """
    Compute the scaled dot-product attention.
    Args:
        query: Queries of shape (batch_size, num_heads, num_patches, head_dim)
        key: Keys of shape (batch_size, num_heads, num_patches, head_dim)
        value: Values of shape (batch_size, num_heads, num_patches, head_dim)
        mask: Optional mask of shape (batch_size, num_heads, num_patches, num_patches)
    Returns:
        output: Attention output of shape (batch_size, num_heads, num_patches, head_dim)
        attention: Attention weights of shape (batch_size, num_heads, num_patches, num_patches)
    """
    d_k = query.size(-1)  # head_dim
    scores = torch.matmul(query, key.transpose(-2, -1)) / (d_k**0.5)
    if use_mask:
        mask = torch.triu(torch.ones_like(scores), diagonal=1)
        scores = scores.masked_fill(mask == 1, float("-inf"))
    attention = F.softmax(scores, dim=-1)
    output = torch.matmul(attention, value)  # Weighted sum of values
    return output, attention


class MultiHeadAttention(nn.Module):
    def __init__(self, x_dim, y_dim, output_dim, num_heads=8):
        super(MultiHeadAttention, self).__init__()
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.head_dim = output_dim // num_heads

        # Linear layers for Q, K, V
        self.query_fc = nn.Linear(x_dim, output_dim)
        self.key_fc = nn.Linear(y_dim, output_dim)
        self.value_fc = nn.Linear(y_dim, output_dim)

        # Output linear layer
        self.fc_out = nn.Linear(output_dim, output_dim)

    def forward(self, x, y, use_mask=False):
        # x shape: (batch_size, num_patches, x_dim) - Query
        # y shape: (batch_size, num_patches, y_dim) - Key and Value

        batch_size = x.size(0)

        # Linear projections
        query = self.query_fc(x)
        key = self.key_fc(y)
        value = self.value_fc(y)

        # Split into multiple heads
        # [batch_size, num_heads, num_patches, head_dim]
        query = query.view(batch_size, -1, self.num_heads, self.head_dim).transpose(
            1, 2
        )
        key = key.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, self.num_heads, self.head_dim).transpose(
            1, 2
        )

        # Compute attention per head
        output, _ = scaled_dot_product_attention(query, key, value, use_mask)

        # Concatenate heads
        output = (
            output.transpose(1, 2).contiguous().view(batch_size, -1, self.output_dim)
        )

        # Final linear layer
        # (batch_size, num_patches, num_heads * head_dim) -> (batch_size, num_patches, output_dim)
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
        self.attention = MultiHeadAttention(
            x_dim=model_dim, y_dim=model_dim, output_dim=model_dim, num_heads=num_heads
        )
        self.ffn = FeedForwardNetwork(model_dim, ff_dim)
        self.layernorm1 = nn.LayerNorm(model_dim)  # LayerNorm for attention output
        self.layernorm2 = nn.LayerNorm(model_dim)  # LayerNorm for FFN output

    def forward(self, x):
        # x shape: (batch_size, num_patches, model_dim)

        # Attention
        attention_output = self.attention(x, x)
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
        self.num_patches = (image_size // patch_size) ** 2
        self.encoder = TransformerEncoder(model_dim, num_heads, ff_dim, num_layers, self.num_patches)
        self.fc = nn.Linear(model_dim, 10)  # classifier

    def forward(self, x):
        # x shape: (batch_size, num_channels, height, width)
        x = self.input_projection(x)  # [batch_size, num_patches, model_dim]
        x = self.encoder(x)
        x = x.mean(dim=1)  # Global average pooling
        x = self.fc(x)  # [batch_size, 10]
        return x


class DecoderLayer(nn.Module):
    def __init__(
        self, image_embedding_dim, text_embedding_dim, num_heads=8, ff_dim=512
    ):
        super(DecoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(
            x_dim=text_embedding_dim,
            y_dim=text_embedding_dim,
            output_dim=text_embedding_dim,
            num_heads=num_heads,
        )
        self.layernorm1 = nn.LayerNorm(text_embedding_dim)
        self.cross_attention = MultiHeadAttention(
            x_dim=text_embedding_dim,
            y_dim=image_embedding_dim,
            output_dim=text_embedding_dim,
            num_heads=num_heads,
        )
        self.layernorm2 = nn.LayerNorm(text_embedding_dim)
        self.ffn = FeedForwardNetwork(text_embedding_dim, ff_dim)
        self.layernorm3 = nn.LayerNorm(text_embedding_dim)

    def forward(self, decoder_input, encoder_output):
        # decoder_input shape: (batch_size, target_length, text_embedding_dim)
        # encoder_output shape: (batch_size, source_length, image_embedding_dim)

        self_attention_output = self.self_attention(x=decoder_input, y=decoder_input, use_mask=True)
        x = self.layernorm1(
            decoder_input + self_attention_output
        )  # Residual connection

        cross_attention_output = self.cross_attention(x, y=encoder_output)
        x = self.layernorm2(x + cross_attention_output)  # Residual connection

        ffn_output = self.ffn(x)
        x = self.layernorm3(x + ffn_output)
        return x


class TransformerDecoder(nn.Module):
    def __init__(
        self,
        image_embedding_dim,
        text_embedding_dim,
        len_seq,
        num_heads=8,
        ff_dim=512,
        num_layers=4,
    ):
        super(TransformerDecoder, self).__init__()
        self.layers = nn.ModuleList(
            [
                DecoderLayer(image_embedding_dim, text_embedding_dim, num_heads, ff_dim)
                for _ in range(num_layers)
            ]
        )
        self.positional_encoding = nn.Parameter(torch.randn(1, len_seq, text_embedding_dim))

    def forward(self, decoder_input, encoder_output):
        # decoder_input shape: (batch_size, target_length, text_embedding_dim)
        # encoder_output shape: (batch_size, source_length, image_embedding_dim)

        decoder_input = decoder_input + self.positional_encoding
        for layer in self.layers:
            decoder_input = layer(decoder_input, encoder_output)
        return decoder_input


class EncoderDecoderTransformerMNIST(nn.Module):
    def __init__(
        self,
        num_classes, # 13 - 10 digits + 3 special tokens
        image_size=56,  # 2x2 images
        patch_size=14,
        grid_size=2,
        image_embedding_dim=64,
        text_embedding_dim=16,
    ):
        super(EncoderDecoderTransformerMNIST, self).__init__()
        self.num_patches = (image_size // patch_size) ** 2
        self.seq_len = grid_size**2 + 1  # 2x2 grid of images + 1 for <sos> token
        self.input_projection = InputProjection(image_size, patch_size, image_embedding_dim)
        self.encoder = TransformerEncoder(image_embedding_dim, max_length=self.num_patches)
        self.text_embedding = nn.Embedding(num_classes, text_embedding_dim)
        self.decoder = TransformerDecoder(image_embedding_dim, text_embedding_dim, self.seq_len)
        self.fc = nn.Linear(text_embedding_dim, num_classes)  # classifier

    def forward(self, x, labels):
        # x shape: (batch_size, num_channels, height, width) - 2x2 grid of images
        # labels shape: (batch_size, seq_len) - sequence of labels
        x = self.input_projection(x)
        encoder_output = self.encoder(x)

        target = self.text_embedding(labels)
        decoder_output = self.decoder(target, encoder_output)
        x = self.fc(decoder_output)
        return x
