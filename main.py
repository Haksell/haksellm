import re

import tiktoken

TOKEN_END = "<|endoftext|>"  # from tiktoken
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


def main():
    with open("the-verdict.txt", encoding="utf-8") as f:
        raw_text = f.read()
    print("Total number of character:", len(raw_text))
    print(raw_text[:99])

    preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
    preprocessed = [item.strip() for item in preprocessed if item.strip()]
    print(preprocessed[:30])

    vocab = sorted(set(preprocessed))
    vocab += [TOKEN_END, TOKEN_UNK]
    tokenizer = Tokenizer(vocab)
    print(len(vocab))

    text1 = "Hello, do you like tea?"
    text2 = "In the sunlit terraces of someunknownPlace."
    text = f"{text1} {TOKEN_END} {text2}"
    print(text)
    encoded = tokenizer.encode(text)
    print(encoded)
    decoded = tokenizer.decode(encoded)
    print(decoded)

    tokenizer = tiktoken.get_encoding("gpt2")
    encoded = tokenizer.encode(text, allowed_special={TOKEN_END})
    print(encoded)
    decoded = tokenizer.decode(encoded)
    print(decoded)

    with open("the-verdict.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()
    enc_text = tokenizer.encode(raw_text)
    print(len(enc_text))

    context_size = 4
    enc_sample = enc_text[50:]
    for i in range(1, context_size + 1):
        context = enc_sample[:i]
        desired = enc_sample[i]
        print(
            tokenizer.decode(context),
            f"({context})",
            "---->",
            tokenizer.decode([desired]),
            f"({desired})",
        )


if __name__ == "__main__":
    main()
