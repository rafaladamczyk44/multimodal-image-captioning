from dataset import FlickrDataset
from torchvision import transforms


# Change res to 256x256, change img to tensor
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

dataset = FlickrDataset(root_dir='dataset/flickr30k_images',
                        captions_file='dataset/captions_formatted.txt',
                        transformations=transform)