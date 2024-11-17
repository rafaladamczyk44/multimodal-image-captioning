from nltk import word_tokenize, FreqDist
from sympy import sequence


# TODO: Add padding

class Tokenizer:
    def __init__(self, file):
        self.file = file
        # Build a vocab out of given file directly after initialization
        self.vocab = self._build_vocab()

    def __len__(self):
        return len(self.vocab)

    def _build_vocab(self):
        """Moving tokenize() method functionality here,
        as this method will build vocabulary, tokenize() will be used for tokenizing captions"""
        with open(self.file, 'r') as f:
            lines = f.readlines()
            just_text = []
            for line in lines:
                just_text.append(line.split('\t')[1])

        f.close()
        tokenized_sentences = []

        for line in just_text:
            # Split sentences into words - tokens
            tokens = word_tokenize(line)
            tokenized_sentences.append(tokens)

        all_tokens = [token for sentence in tokenized_sentences for token in sentence]

        # Add tokens values based on the frequency
        freq_dist = FreqDist(all_tokens)

        # Special tokens for padding, start/end of sentence and unknown tokens
        # 0, 1, 2, 3 indexes for special tokens
        special_tokens = ['<PAD>', '<SOS>', '<EOS>', '<UNK>']

        # Start a vocab dict.
        # Add special tokens
        vocab = {token: idx for idx, token in enumerate(special_tokens)}

        # Add words to vocab
        for word in freq_dist:
            # Avoid duplicates
            if word not in vocab:
                vocab[word] = len(vocab)

        return vocab

    def tokenize(self, caption:str):
        """
        This method returns tokenized caption seq.
        """
        tokens = word_tokenize(caption)
        # Get the token number from dict or unknown
        token_ind = [self.vocab.get(token, self.vocab['<UNK>']) for token in tokens]
        # Return a sentence SOS + tokens + EOS
        return [self.vocab['<SOS>']] + token_ind + [self.vocab['<EOS>']]

    def pad_sequence(self, seq, max_length):
        """
        This method pads the tokenized caption seq.
        """
        pad_index = self.vocab['<PAD>']
        if len(seq) < max_length:
            # SEQ + PAD + PAD + PAD...
            seq += [pad_index] * (max_length - len(seq))
        else:
            seq = seq[:max_length]
        return seq
