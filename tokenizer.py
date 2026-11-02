import re

TOKEN_UNK = "<|unk|>"


class Tokenizer:
    def __init__(self, vocab):
        self.int_to_tok = vocab
        self.tok_to_int = {s: i for i, s in enumerate(vocab)}

    def encode(self, text):
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        unk_id = self.tok_to_int[TOKEN_UNK]
        ids = [self.tok_to_int.get(s, unk_id) for s in preprocessed]
        return ids

    def decode(self, ids):
        text = " ".join([self.int_to_tok[i] for i in ids])
        text = re.sub(r'\s+([,.:;?!"()\'])', r"\1", text)
        return text.strip()
