## Multimodal Image Captioning
Author: Rafał Adamczyk

The goal of this project is to create a multimodal AI agent capable of providing a caption for presented image

### Data
Dataset used in project: https://www.kaggle.com/datasets/adityajn105/flickr30k/data
It can be downloaded with:
```
path = kagglehub.dataset_download("adityajn105/flickr30k")
```
### Model Architecture
- For feature extraction I'm using ResNet model (https://pytorch.org/hub/pytorch_vision_resnet/)
- Extracted features are passed together with tokenized image captions to Transformer
- Transformer learns how to generate captions


### Project history:

- I have first started with single word tokenization, but this turned out to be quite inefficient, as it was consuming whole ram of my laptop
- Then I modified the Tokenizer to tokenize single characters reducing vocabulary from 23k to 67, which was great performance-wise, but the model couldn't converge

### TODO:
- Use WordPiece for tokenization