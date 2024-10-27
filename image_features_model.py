from dataset import FlickrDataset
from torchvision import transforms
from torchvision.models import resnet34

# TODO: Tokenize captions
# TODO: Create a model
# TODO: Train model on the data
"""
ResNet docs PyTorch: https://pytorch.org/hub/pytorch_vision_resnet/
"""

model = resnet34()

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


print(dataset.__getitem__(1))
