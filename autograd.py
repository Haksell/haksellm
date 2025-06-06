import torch
import torch.nn as nn
import torch.optim as optim

inputs = torch.tensor(
    [
        [22, 25],
        [25, 35],
        [47, 80],
        [52, 95],
        [46, 82],
        [56, 90],
        [23, 27],
        [30, 50],
        [40, 60],
        [39, 57],
        [53, 95],
        [48, 88],
    ],
    dtype=torch.float32,
)

labels = torch.tensor(
    [[0], [0], [1], [1], [1], [1], [0], [1], [1], [0], [1], [1]], dtype=torch.float32
)

model = nn.Sequential(
    nn.Linear(inputs.shape[1], 3),
    nn.ReLU(),
    nn.Linear(3, 1),
    nn.Sigmoid(),
)
optimizer = optim.SGD(model.parameters(), lr=0.001)
criterion = nn.BCELoss()

for _ in range(500):
    optimizer.zero_grad()
    loss = criterion(model(inputs), labels)
    loss.backward()
    optimizer.step()

print(list(model.parameters()))
