from tokenizers import Tokenizer
from tokenizers.models import WordPiece
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import WordPieceTrainer

file = 'dataset/captions_formatted.txt'

with open(file, 'r') as f:
    lines = f.readlines()
captions = [line.split('\t')[1].strip() for line in lines]

# Initialize WordPiece tokenizer
tokenizer = Tokenizer(WordPiece(unk_token="<UNK>"))
tokenizer.pre_tokenizer = Whitespace()

# Train the tokenizer
trainer = WordPieceTrainer(
    vocab_size=5000,
    special_tokens=['<PAD>', '<SOS>', '<EOS>', '<UNK>']
)
tokenizer.train_from_iterator(captions, trainer=trainer)

# Save the tokenizer to a file
tokenizer.save("wordpiece_tokenizer.json")

