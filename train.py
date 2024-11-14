from dataset import FlickrDataset
from tokenizer import Tokenizer

from torchvision import transforms
from torchvision.models import resnet34

file = 'dataset/captions_formatted.txt'

vocab = Tokenizer(file).tokenize()

# Change res to 256x256, change img to tensor
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

dataset = FlickrDataset(root_dir='dataset/flickr30k_images',
                        captions_file='dataset/captions_formatted.txt',
                        transformations=transform)
