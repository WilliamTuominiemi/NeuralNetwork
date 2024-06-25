import numpy as np
import pandas as pd

import torch
import torch.nn as nn

torch.manual_seed(42)

class OneHidden(nn.Module):
    def __init__(self):
        super(OneHidden, self).__init__()
        self.layer1 = nn.Linear(2, 4)
        self.layer2 = nn.Linear(4, 1)
        
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x


model = OneHidden()

X = torch.tensor([3,4.5], dtype=torch.float32)

predictions = model(X)

print(predictions)