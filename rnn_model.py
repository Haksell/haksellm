import math
import os
import random
import re
import tarfile
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, IterableDataset
from transformers import AutoTokenizer
import urllib.request


class RNNUnit(nn.Module):
    def __init__(self, emb_dim):
        super().__init()
        self.uh = nn.Parameter(torch.randn(emb_dim, emb_dim))
        self.wh = nn.Parameter(torch.randn(emb_dim, emb_dim))
        self.b = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x, h):
        return torch.tanh(x @ self.wh + h @ self.uh + self.b)


class RNN(nn.Module):
    def __init__(self, emb_dim, num_layers):
        super().__init()
        self.emb_dim = emb_dim
        self.num_layers = num_layers
        self.rnn_units = nn.ModuleList([RNNUnit(emb_dim) for _ in range(num_layers)])

    def forward(self, x):
        batch_size, seq_len, emb_dim = x.shape
        h_prev = [
            torch.zeros(batch_size, emb_dim, device=x.device)
            for _ in range(self.num_layers)
        ]
        outputs = []
        for input_t in torch.unbind(x, dim=1):
            for layer, rnn_unit in enumerate(self.rnn_units):
                input_t = h_prev[layer] = rnn_unit(input_t, h_prev[layer])
            outputs.append(input_t)
        return torch.stack(outputs, dim=1)


class RNNModel(nn.Module):
    def __init__(self, vocab_size, emb_dim, num_layers, padding_idx):
        super().__init()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=padding_idx)
        self.rnn = RNN(emb_dim, num_layers)
        self.fc = nn.Linear(emb_dim, vocab_size)

    def forward(self, x):
        embeddings = self.embedding(x)
        rnn_output = self.rnn(embeddings)
        logits = self.fc(rnn_output)
        return logits


# https://github.com/aburkov/theLMbook/blob/main/news_RNN_language_model.ipynb
class IterableTextDataset(IterableDataset):
    def __init__(self, file_path, tokenizer, max_length):
        self.file_path = file_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self._count_sentences()

    def __iter__(self):
        # Open file in read mode with UTF-8 encoding
        with open(self.file_path, "r", encoding="utf-8") as f:
            # Process each line (sentence) in the file
            for line in f:
                # Remove leading/trailing whitespace
                sentence = line.strip()
                # Replace all numbers with ### placeholder
                # This reduces vocabulary size and helps model generalize
                sentence = re.sub(r"\d+", "###", sentence)

                # Convert sentence to token IDs
                encoded_sentence = self.tokenizer.encode(
                    sentence, max_length=self.max_length, truncation=True
                )

                # Only use sequences with at least 2 tokens
                # (need at least one input and one target token)
                if len(encoded_sentence) >= 2:
                    # Input is all tokens except last
                    input_seq = encoded_sentence[:-1]
                    # Target is all tokens except first
                    target_seq = encoded_sentence[1:]
                    # Convert to PyTorch tensors and yield
                    yield (
                        torch.tensor(input_seq, dtype=torch.long),
                        torch.tensor(target_seq, dtype=torch.long),
                    )

    def __len__(self):
        return self._num_sentences

    def _count_sentences(self):
        print(f"Counting sentences in {self.file_path}...")
        with open(self.file_path, "r", encoding="utf-8") as f:
            self._num_sentences = sum(1 for _ in f)
        print(f"Found {self._num_sentences} sentences in {self.file_path}.")


EMBEDDING_DIMENSION = 128
NUM_LAYERS = 2
BATCH_SIZE = 128
LEARNING_RATE = 0.001
NUM_EPOCHS = 1
CONTEXT_SIZE = 30
FILENAME = "news.tar.gz"


# https://github.com/aburkov/theLMbook/blob/main/news_RNN_language_model.ipynb
def download_and_prepare_data(url, batch_size, tokenizer, context_size):
    # Download file
    if not os.path.exists(FILENAME):
        print(f"Downloading dataset from {url}...")
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req) as response:
            with open(FILENAME, "wb") as out_file:
                out_file.write(response.read())
        print("Download completed.")
    else:
        print(f"{FILENAME} already downloaded.")

    # Extract file
    data_dir = os.path.join(os.path.dirname(FILENAME), "news")
    train_path = os.path.join(data_dir, "train.txt")
    test_path = os.path.join(data_dir, "test.txt")
    if os.path.exists(train_path) and os.path.exists(test_path):
        print("Data files already extracted.")
        return train_path, test_path
    print("\nListing archive contents:")
    with tarfile.open(FILENAME, "r:gz") as tar:
        for member in tar.getmembers():
            print(f"Archive member: {member.name}")

        print("\nExtracting files...")
        tar.extractall(".")
    if not (os.path.exists(train_path) and os.path.exists(test_path)):
        raise FileNotFoundError(
            "Required files not found in the archive. Please check the paths above."
        )
    print("Extraction completed.")

    # Create datasets
    train_dataset = IterableTextDataset(train_path, tokenizer, context_size)
    test_dataset = IterableTextDataset(test_path, tokenizer, context_size)
    print(f"Training sentences: {len(train_dataset)}")
    print(f"Test sentences: {len(test_dataset)}")

    # Create dataloaders
    def collate_fn(batch):
        input_seqs, target_seqs = zip(*batch)
        input_padded = nn.utils.rnn.pad_sequence(
            input_seqs, batch_first=True, padding_value=tokenizer.pad_token_id
        )
        target_padded = nn.utils.rnn.pad_sequence(
            target_seqs, batch_first=True, padding_value=tokenizer.pad_token_id
        )
        return input_padded, target_padded

    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, collate_fn=collate_fn, num_workers=0
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=batch_size, collate_fn=collate_fn, num_workers=0
    )
    return train_dataloader, test_dataloader


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def initialize_weights(model):
    for _, param in model.named_parameters():
        if param.dim() > 1:
            nn.init.xavier_uniform_(param)
        else:
            nn.init.uniform_(param)


def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3.5-mini-instruct")
    vocab_size = len(tokenizer)
    train_loader, test_loader = download_and_prepare_data(
        "https://www.thelmbook.com/data/news", BATCH_SIZE, tokenizer, CONTEXT_SIZE
    )

    rnn_model = RNNModel(
        vocab_size, EMBEDDING_DIMENSION, NUM_LAYERS, tokenizer.pad_token_id
    )
    initialize_weights(rnn_model)
    rnn_model.to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    optimizer = torch.optim.AdamW(rnn_model.parameters(), lr=LEARNING_RATE)

    for _ in range(NUM_EPOCHS):
        rnn_model.train()
        for input_seq, target_seq in train_loader:
            input_seq = input_seq.to(device)
            target_seq = target_seq.to(device)
            optimizer.zero_grad()
            output = rnn_model(input_seq)
            target_size = math.prod(input_seq.shape)
            loss = criterion(
                output.reshape(target_size, vocab_size),
                target_seq.reshape(target_size),
            )
            loss.backward()
            optimizer.step()


if __name__ == "__main__":
    main()
