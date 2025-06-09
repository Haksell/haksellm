import torch
import torch.nn as nn

vocab_size = 5
emb_dim = 3
embedding_layer = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
token_indices = torch.tensor([0, 2, 4])
embeddings = embedding_layer(token_indices)
print(embeddings)
