import torch
from models.cnn_model import CNNModel

model = CNNModel()

x = torch.randn(1, 1, 224, 224)
output = model(x)

print(output)