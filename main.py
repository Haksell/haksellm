import re


class Tokenizer:
    def __init__(self, vocab):
        self.int_to_tok = vocab
        self.tok_to_int = {s: i for i, s in enumerate(vocab)}

    def encode(self, text):
        preprocessed = re.split(r'([,.?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        ids = [self.tok_to_int[s] for s in preprocessed]
        return ids

    def decode(self, ids):
        text = " ".join([self.int_to_tok[i] for i in ids])
        text = re.sub(r'\s+([,.?!"()\'])', r"\1", text)
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
    tokenizer = Tokenizer(vocab)

    text = """"It's the last he painted, you know,"
    Mrs. Gisburn said with pardonable pride."""
    ids = tokenizer.encode(text)
    print(ids)
    print(tokenizer.decode(ids))


if __name__ == "__main__":
    main()
