import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class FlickrDataset(Dataset):
    """
    Since Pytorch flickr dataset class doesn't work, this custom does its jon
    input:
        root_dir:str - folder with images
        captions_file:str - path to file with captions (img /t img_caption)
        transforms: transformations to be applied on a sample
    """
    def __init__(self, root_dir, captions_file, transformations=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_captions = []

        # Read the captions file
        with open(captions_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                # Verify if both are present - img and caption
                if len(parts) == 2:
                    self.image_captions.append((parts[0], parts[1]))

    def __len__(self):
        return len(self.image_captions)

    def __getitem__(self, idx):
        img_name, caption = self.image_captions[idx]
        img_path = os.path.join(self.root_dir, img_name)

        # Open the image
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, caption


# Change res to 256x256, change img to tensor
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

# dataset = FlickrDataset(root_dir='dataset/flickr30k_images',
#                         captions_file='dataset/captions_formatted.txt',
#                         transformations=transform)


