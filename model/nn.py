import torch, torch.nn as nn


class TextCNN(nn.Module):
    def __init__(self, vocab_sz, emb_dim, pad_idx):
        super().__init__()
        self.embedding = nn.Embedding(vocab_sz, emb_dim, padding_idx=pad_idx)
        self.conv = nn.Conv1d(emb_dim, 128, kernel_size=3)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.drop = nn.Dropout(0.5)
        self.fc = nn.Linear(128, 1)

    def forward(self, x):  # x: (B, L)
        x = self.embedding(x).transpose(1, 2)  # (B, E, L)
        x = self.pool(self.relu(self.conv(x))).squeeze(2)  # (B, 128)
        x = self.drop(x)
        return torch.sigmoid(self.fc(x)).squeeze(1)
