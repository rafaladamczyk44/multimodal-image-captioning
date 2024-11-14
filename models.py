import torch
import torch.nn as nn
from torchvision.models import resnet34

class Encoder(nn.Module):
    """
    Encoder is based on pre-trained ResNet model.
    I'm removing the last layer as I need to extract features only
    """
    def __init__(self):
        super().__init__()
        self.resnet = resnet34(pretrained=True)
        # Last layer is linear for classification, don't need it here, just features of the passed image
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-2])

    def forward(self, x):
        # Return features
        # Not pooling if to keep the grid for attention based decoder
        return self.resnet(x)  # Shape: (batch_size, 512, 7, 7)


class Decoder(nn.Module):
    """
    Decoder takes the output of Encoder class, which is a matrix of extracted features from an image
    It will be trained on vocabulary build in the project to generate a description of provided image
    So changing extracted features into attention matrix output
    """
    def __init__(self, vocab_size, d_model=512, n_heads=8, num_layers=6, max_seq_len=48):
        super(Decoder).__init__()
        self.d_model = d_model
        # Embedding transforms tokens into dense layers
        self.embedding = nn.Embedding(vocab_size, d_model)
        # Positional encoding for learns positions and context
        self.positional_encoding = self._generate_positional_encoding(max_seq_len, d_model)

        # Transformer decoder
        self.transformer_decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=n_heads)
        self.transformer_decoder = nn.TransformerDecoder(self.transformer_decoder_layer, num_layers=num_layers)

        # Fully connected linear layer for output
        self.linear = nn.Linear(d_model, vocab_size)

    def _generate_positional_encoding(self, max_seq_len, d_model):
        """Create positional encoding matrix"""
        # Create an empty matrix
        positional_encoding = torch.zeros(max_seq_len, d_model)
        # Create a column vector representing words order
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        # Sin and cos freq. control
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        # From attention arch.; putting embeddings on sin and cos
        positional_encoding[:, 0::2] = torch.sin(position * div_term)
        positional_encoding[:, 1::2] = torch.cos(position * div_term)
        # Add extra dimension
        positional_encoding = positional_encoding.unsqueeze(0)
        return positional_encoding  # Shape: (1, max_seq_len, d_model)

    def forward(self, tgt, memory):
        """
        tgt: tokenized caption, shape: (seq_len, batch_size)
        memory: image features extracted by encoder, shape: (batch_size, 49, d_model)
        """

        # Embed the target (caption) and add positional encoding
        tgt_embedded = self.embedding(tgt) * torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32))
        tgt_embedded = tgt_embedded + self.positional_encoding[:, :tgt.size(0), :].to(tgt.device)

        # Transformer decoder expects (seq_len, batch, d_model), so no transpose needed here
        tgt_mask = nn.Transformer().generate_square_subsequent_mask(tgt.size(0)).to(tgt.device)

        output = self.transformer_decoder(tgt_embedded, memory, tgt_mask=tgt_mask)
        output = self.linear(output)
        return output # Shape: (seq_len, batch_size, vocab_size)
