from nltk import word_tokenize, FreqDist
import string
from sympy import sequence


class Tokenizer:
    def __init__(self, file):
        self.file = file
        # Build a vocab out of given file directly after initialization
        self.vocab = self._build_vocab()

    def __len__(self):
        return len(self.vocab)

    def _build_vocab(self):
        """
        Build vocabulary from text file
        """
        with open(self.file, 'r') as f:
            lines = f.readlines()
            just_text = [line.split('\t')[1] for line in lines]
        f.close()

        # Special tokens for padding, start/end of sentence and unknown tokens
        # 0, 1, 2, 3 indexes for special tokens
        special_tokens = ['<PAD>', '<SOS>', '<EOS>', '<UNK>']
        letters = set(''.join(just_text))

        allowed_letters = set(letters) - set(string.punctuation)
        allowed_letters = {char for char in allowed_letters if char.isalnum()}

        vocab = {token: idx for idx, token in enumerate(special_tokens)}

        for char in sorted(allowed_letters):
            if char not in vocab:
                vocab[char] = len(vocab)

        return vocab

    def tokenize(self, caption:str):
        """
        Tokenize a caption
        """
        token_ind = [self.vocab.get(char, self.vocab['<UNK>']) for char in caption if char.isalnum()]
        # Return a sentence SOS + tokens + EOS
        return [self.vocab['<SOS>']] + token_ind + [self.vocab['<EOS>']]

    def pad_sequence(self, seq, max_length):
        """
        Pad the tokenized caption to match max_seq_len
        """
        pad_index = self.vocab['<PAD>']
        if len(seq) < max_length:
            # SEQ + PAD + PAD + PAD...
            seq += [pad_index] * (max_length - len(seq))
        else:
            seq = seq[:max_length]
        return seq
