import torch
import torch.nn as nn
from torch.cuda import device
from torch.optim import Adam, AdamW
from torchvision import transforms
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split

from tokenizers import Tokenizer

from dataset import FlickrDataset
from tokenizer import Tokenizer, tokenizer
from models import Encoder, Decoder

def collate_fn(batch):
    images, captions = zip(*batch)

    images = torch.stack(images)
    captions = pad_sequence([torch.tensor(caption) for caption in captions], batch_first=True,
                            padding_value=tokenizer.token_to_id('<PAD>'))

    return images, captions

def train(dataloader, encoder, decoder, criterion, optimizer):
    encoder.train()
    decoder.train()

    for batch_idx, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device).long()

        img_features = encoder(X)
        pred = decoder(y, img_features)

        loss = criterion(pred.permute(0, 2, 1).reshape(-1, vocab_size), y.reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % 100 == 0:
            print(f'loss: {loss.item():>7f}  [{batch_idx * len(X):>5d}]')

# Training parameters
epochs = 50
lr = 0.001
batch_size = 16
max_seq_len = 32

file = 'dataset/captions_formatted.txt'
tokenizer_pretrained = 'wordpiece_tokenizer.json'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Models setup
tokenizer = Tokenizer.from_file(tokenizer_pretrained)
vocab_size = tokenizer.get_vocab_size()
enc_model = Encoder()
dec_model = Decoder(vocab_size=vocab_size, max_seq_len=max_seq_len)
loss = nn.CrossEntropyLoss(ignore_index=tokenizer.get_vocab()['<PAD>'])
optimizer = Adam(list(enc_model.parameters()) + list(dec_model.parameters()), lr=lr)

# Load dataset
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
dataset = FlickrDataset(root_dir='dataset/flickr30k_images',
                        captions_file=file,
                        tokenizer=tokenizer,
                        max_seq_len=max_seq_len,
                        transformations=transform)

# Train/test split
indices = list(range(vocab_size))
train_indices, test_indices = train_test_split(indices, test_size=0.2, random_state=42)

# Create dataloader
train_sampler = SubsetRandomSampler(train_indices)
train_dataloader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler, collate_fn=collate_fn)

for epoch in range(epochs):
    print(f'Epoch {epoch+1}')
    train(train_dataloader, enc_model, dec_model, criterion=loss, optimizer=optimizer)

torch.save(dec_model.state_dict(), 'models/decoder.pt')
