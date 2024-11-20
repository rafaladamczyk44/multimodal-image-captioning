import torch
import torch.nn as nn
from torch.optim import Adam, AdamW
from torchvision import transforms
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split

from dataset import FlickrDataset
from tokenizer import Tokenizer
from models import Encoder, Decoder

def collate_fn(batch):
    images, captions = zip(*batch)
    images = torch.stack(images)

    # Tokenize and pad captions
    captions = [torch.tensor(caption, dtype=torch.long) for caption in captions]
    captions = pad_sequence(captions, batch_first=True, padding_value=tokenizer.vocab['<PAD>'])
    return images, captions

def train(dataloader, encoder, decoder, criterion, optimizer):
    encoder.train()
    decoder.train()
    accumulation_steps = 4

    for batch_idx, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device).long()
        # Get features
        img_features = encoder(X)
        # Get decoder predictions
        pred = decoder(y, img_features)
        # Calculate loss
        # loss = criterion(pred.view(-1, vocab_size), y.view(-1))
        loss = criterion(pred.permute(0, 2, 1).reshape(-1, vocab_size), y.reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % 100 == 0:
            print(f'loss: {loss.item():>7f}  [{batch_idx * len(X):>5d}]')

file = 'dataset/captions_formatted.txt'
device = torch.device('cpu')

# Change res to 256x256, change img to tensor
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    # transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

"""Define all needed classes"""
# Vocabulary + tokenizer
tokenizer = Tokenizer(file)
# Get dataset size
vocab_size = len(tokenizer.vocab)
# print(vocab_size)
indices = list(range(vocab_size))

# Dataset with images and captions
dataset = FlickrDataset(root_dir='dataset/flickr30k_images',
                        captions_file='dataset/captions_formatted.txt',
                        tokenizer=tokenizer,
                        max_seq_len=64,
                        transformations=transform)

train_indices, test_indices = train_test_split(indices, test_size=0.2, random_state=42)

# Create samplers
train_sampler = SubsetRandomSampler(train_indices)
# test_sampler = SubsetRandomSampler(test_indices)

# Create DataLoader with samplers
train_dataloader = DataLoader(dataset, batch_size=32, sampler=train_sampler, collate_fn=collate_fn)
# test_dataloader = DataLoader(dataset, batch_size=4, sampler=test_sampler, collate_fn=collate_fn)

enc_model = Encoder()
dec_model = Decoder(vocab_size=vocab_size)

# Training parameters
epochs = 50
lr = 0.01
loss = nn.CrossEntropyLoss(ignore_index=tokenizer.vocab['<PAD>'])

# Optimizer
optimizer = Adam(list(enc_model.parameters()) + list(dec_model.parameters()), lr=lr)


for epoch in range(epochs):
    print(f'Epoch {epoch+1}')
    train(train_dataloader, enc_model, dec_model, criterion=loss, optimizer=optimizer)
