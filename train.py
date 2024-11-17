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
    captions = [torch.tensor(caption, dtype=torch.long) for caption in captions]
    captions = pad_sequence(captions, batch_first=True, padding_value=tokenizer.vocab['<PAD>'])
    return images, captions

def train(dataloader, encoder, decoder, criterion, optimizer):
    encoder.train()
    decoder.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device).long()
        print(X.shape, X.dtype)
        print(y.shape, y.dtype)

        # Get features
        img_features = encoder(X)

        # Get decoder predictions
        pred = decoder(img_features, y)

        loss = criterion(pred.view(-1, vocab_size), y.view(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            print(f'loss: {loss.item():>7f}  [{batch * len(X):>5d}]')

file = 'dataset/captions_formatted.txt'
device = torch.device('cpu')

# Change res to 256x256, change img to tensor
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

"""Define all needed classes"""
# Vocabulary + tokenizer
tokenizer = Tokenizer(file)

# Dataset with images and captions
dataset = FlickrDataset(root_dir='dataset/flickr30k_images',
                        captions_file='dataset/captions_formatted.txt',
                        tokenizer=tokenizer,
                        max_seq_len=20,
                        transformations=transform)


# Get dataset size
vocab_size = tokenizer.__len__()
indices = list(range(vocab_size))

# Split indices
train_indices, test_indices = train_test_split(indices, test_size=0.2, random_state=42)

# Create samplers
train_sampler = SubsetRandomSampler(train_indices)
test_sampler = SubsetRandomSampler(test_indices)

# Create DataLoader with samplers
train_dataloader = DataLoader(dataset, batch_size=32, sampler=train_sampler, collate_fn=collate_fn)
test_dataloader = DataLoader(dataset, batch_size=32, sampler=test_sampler, collate_fn=collate_fn)

# Encoder model to get features
enc_model = Encoder()

# Decoder to train and generate new captions
dec_model = Decoder(vocab_size=vocab_size)

# Testing tokenizer
# image, caption = dataset[1]
# print(vocab.tokenize(caption))
# print(vocab.tokenize('Dog walking in park'))

# Training parameters
epochs = 50
lr = 0.01
# Using cross entropy loss as it's supposed to be good for language model
# Ignoring pad tokens
loss = nn.CrossEntropyLoss(ignore_index=tokenizer.vocab['<PAD>'])

# Optimizer
optimizer = Adam(list(enc_model.parameters()) + list(dec_model.parameters()), lr=lr)

"""
Training process:
Split into train/test (8:2)
Make batches out of images and corresponding captions

Forward pass:
    1. Pass a batch of images through encoder and captions through a tokenizer
    2. Pass this batch to decoder 
    3. Calculate decoder's loss
    
Backward
Eval
Save
"""
for epoch in range(epochs):
    print(f'Epoch {epoch}')
    train(train_dataloader, enc_model, dec_model, criterion=loss, optimizer=optimizer)
