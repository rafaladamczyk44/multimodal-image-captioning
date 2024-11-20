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

# TODO: Check if vocab size is correctly or if it's making bottleneck here
class Decoder(nn.Module):
    """
    Decoder takes the output of Encoder class, which is a matrix of extracted features from an image
    It will be trained on vocabulary build in the project to generate a description of provided image
    So changing extracted features into attention matrix output
    """
    def __init__(self, vocab_size, d_model=512, n_heads=8, num_layers=8, max_seq_len=64):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.memory_projection = nn.Linear(512, d_model)
        self.positional_encoding = self._generate_positional_encoding(max_seq_len, d_model)
        self.transformer_decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=n_heads)
        self.transformer_decoder = nn.TransformerDecoder(self.transformer_decoder_layer, num_layers=num_layers)
        self.linear = nn.Linear(d_model, vocab_size)

    def _generate_positional_encoding(self, max_seq_len, d_model):
        """Create positional encoding matrix"""
        # Create an empty matrix of size max_length x vocab_size, currently 10x512
        positional_encoding = torch.zeros(max_seq_len, d_model)

        # Create a column vector representing words order
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)

        # Positional encoding scaling
        # value[i] i exp((-i) * ln(10k)/d_model)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))

        # 0, 2, 4.... positions are put on sin, others on cos and scaled
        positional_encoding[:, 0::2] = torch.sin(position * div_term)
        positional_encoding[:, 1::2] = torch.cos(position * div_term)

        # Add extra dimension for batch size
        # So eventually this will be batch_size x max_seq_len x vocab_size
        positional_encoding = positional_encoding.unsqueeze(0)
        return positional_encoding  # Shape: (1, max_seq_len, d_model)

    def forward(self, tgt, img_features):
        # Handle different possible input shapes
        # Handle image features
        if img_features.dim() == 4:
            # Reshape 4D tensor from CNN features
            img_memory = img_features.permute(0, 2, 3, 1)  # (batch_size, 8, 8, 512)
            img_memory = img_memory.reshape(img_memory.size(0), -1, img_memory.size(3))  # (batch_size, 64, 512)
        elif img_features.dim() == 2:
            img_memory = img_features.unsqueeze(1)
        else:
            img_memory = img_features

        # Project image features to model dimension
        img_memory = self.memory_projection(img_memory)  # (batch_size, 64, d_model)
        tgt = tgt.long()

        # Embed the target (caption) and add positional encoding
        tgt_embedded = self.embedding(tgt) #* torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32))
        batch_size, seq_len, _ = tgt_embedded.shape

        positional_encoding = self.positional_encoding[:, :seq_len, :].to(tgt.device)
        tgt_embedded = tgt_embedded + positional_encoding

        mask = nn.Transformer().generate_square_subsequent_mask(batch_size).to(tgt.device)
        # mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1).to(tgt.device)
        # mask = mask.unsqueeze(0).expand(batch_size, -1, -1)

        output = self.transformer_decoder(tgt_embedded, img_memory, tgt_mask=mask)
        output = self.linear(output)
        return output  # Shape: (seq_len, batch_size, vocab_size)
