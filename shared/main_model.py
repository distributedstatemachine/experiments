import torch.nn as nn

class GlobalModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(128, 128)
    def forward(self, x):
        return self.fc(x)
