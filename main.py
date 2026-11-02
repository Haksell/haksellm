import re

import tiktoken
import torch
from torch.utils.data import DataLoader, Dataset

TOKEN_END = "<|endoftext|>"  # from tiktoken


class GPTDataset(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []
        token_ids = tokenizer.encode(txt)
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i : i + max_length]
            target_chunk = token_ids[i + 1 : i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader(
    txt,
    *,
    batch_size=4,
    max_length=256,
    stride=128,
    shuffle=True,
    drop_last=True,
    num_workers=0,
):
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDataset(txt, tokenizer, max_length, stride)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
    )


def main():
    with open("the-verdict.txt", encoding="utf-8") as f:
        raw_text = f.read()
    print("Total number of character:", len(raw_text))
    print(raw_text[:99])

    preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
    preprocessed = [item.strip() for item in preprocessed if item.strip()]
    print(preprocessed[:30])

    text1 = "Hello, do you like tea?"
    text2 = "In the sunlit terraces of someunknownPlace."
    text = f"{text1} {TOKEN_END} {text2}"
    print(text)

    tokenizer = tiktoken.get_encoding("gpt2")
    encoded = tokenizer.encode(text, allowed_special={TOKEN_END})
    print(encoded)
    decoded = tokenizer.decode(encoded)
    print(decoded)
    enc_text = tokenizer.encode(raw_text)
    print(len(enc_text))

    dataloader = create_dataloader(
        raw_text, batch_size=8, max_length=4, stride=4, shuffle=False
    )
    inputs, targets = next(iter(dataloader))
    print(inputs)
    print(targets)

    vocab_size = 6
    embedding_dim = 3
    torch.manual_seed(123)
    embedding_layer = torch.nn.Embedding(vocab_size, embedding_dim)
    print(embedding_layer.weight)
    print(embedding_layer(torch.tensor([2, 3, 5, 1])))


if __name__ == "__main__":
    main()
