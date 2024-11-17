import os
from PIL import Image
from torch.utils.data import Dataset


class FlickrDataset(Dataset):
    """
    Since Pytorch flickr dataset class doesn't work, this custom does its jon
    input:
        root_dir:str - folder with images
        captions_file:str - path to file with captions (img /t img_caption)
        transforms: transformations to be applied on a sample
    """
    def __init__(self, root_dir, captions_file,tokenizer, max_seq_len=20, transformations=None):
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
        img_path = os.path.join(self.root_dir, img_name)

        # Open the image
        image = Image.open(img_path).convert('RGB')

        if self.transformations:
            image = self.transformations(image)

        # Tokenize and pad the caption
        tokenized_caption = self.tokenizer.tokenize(caption)
        padded_caption = self.tokenizer.pad_sequence(tokenized_caption, self.max_seq_len)

        return image, padded_caption
