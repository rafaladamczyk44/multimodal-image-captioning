from tokenizer import Tokenizer

file = 'dataset/captions_formatted.txt'

vocab = Tokenizer(file).tokenize()

print(vocab['cat'])