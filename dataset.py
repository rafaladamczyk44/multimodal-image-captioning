import os
from PIL import Image
from torch.utils.data import Dataset
from tokenizers import Tokenizer

# tokenizer = Tokenizer.from_file('wordpiece_tokenizer.json')

class FlickrDataset(Dataset):
    """
    Since Pytorch flickr dataset class doesn't work, this custom does its jon
    input:
        root_dir:str - folder with images
        captions_file:str - path to file with captions (img /t img_caption)
        transforms: transformations to be applied on a sample
    """
    def __init__(self, root_dir, captions_file,tokenizer, max_seq_len, transformations=None):
        self.root_dir = root_dir
        self.transformations = transformations
        self.image_captions = []
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

        # Read the captions file
        with open(captions_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                # Verify if both are present - img and caption
                if len(parts) == 2:
                    self.image_captions.append((parts[0], parts[1]))

    def __len__(self):
        return len(self.image_captions) / 5

    def __getitem__(self, idx):
        img_name, caption = self.image_captions[idx]
        # print(f"Caption type: {type(caption)}, Caption value: {caption}")
        img_path = os.path.join(self.root_dir, img_name)

        # Open the image
        image = Image.open(img_path).convert('RGB')

        if self.transformations:
            image = self.transformations(image)

        # Tokenize and pad the caption
        tokenized_caption = self.tokenizer.encode(caption).ids

        tokenized_caption = tokenized_caption[:self.max_seq_len]  # truncate if too long
        tokenized_caption += [self.tokenizer.token_to_id('<PAD>')] * (self.max_seq_len - len(tokenized_caption))
        # padded_caption = self.tokenizer.pad_sequence(tokenized_caption, self.max_seq_len)

        return image, tokenized_caption
