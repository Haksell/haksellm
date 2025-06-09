import torch
import torch.nn as nn


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
        ]  # useless to store the array?
        outputs = []
        # for t in range(seq_len):
        #     input_t = x[:, t]
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
