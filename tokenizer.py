from nltk import word_tokenize, FreqDist

class Tokenizer:
    def __init__(self, file):
        self.file = file

    def tokenize(self):
        """
        This method returns vocab dictionary, where all tokens are assigned a frequency num
        """
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
