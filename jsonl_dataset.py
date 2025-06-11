import json
import torch
from torch.utils.data import DataLoader, Dataset


class JSONLDataset(Dataset):
    def __init__(self, file_path):
        self.data = []
        with open(file_path) as f:
            for line in f:
                item = json.loads(line)
                features = [item["feature1"], item["feature2"]]
                label = item["label"]
                self.data.append((features, label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        features, label = self.data[idx]
        features = torch.tensor(features, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.long)
        return features, label


def format_tensor(tensor):
    return "".join(c for c in str(tensor) if not c.isspace())


dataset = JSONLDataset("data/dummy.jsonl")
data_loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)

for epoch in range(5):
    print(f"Epoch #{epoch}")
    for batch_features, batch_labels in data_loader:
        print(f"{format_tensor(batch_features)} -> {format_tensor(batch_labels)}")
