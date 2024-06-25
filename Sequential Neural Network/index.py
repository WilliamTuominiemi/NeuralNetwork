import numpy as np
import pandas as pd
import torch
import torch.nn as nn

torch.manual_seed(42)

model = nn.Sequential(
    nn.Linear(3,8),
    nn.ReLU(),
    nn.Linear(8,4),
    nn.Sigmoid(),
    nn.Linear(4,1)

)

print(model)

apartments_df = pd.read_csv("../streeteasy.csv")

apartments_numpy = apartments_df[['size_sqft', 'bedrooms', 'building_age_yrs']].values

X = torch.tensor(apartments_numpy,dtype=torch.float32)

print(X[:5])

predicted_rent = model(X)

print(predicted_rent[:5])