from models.efficientnet import efficientnet_b7
import torch

#print(torch.hub.list('pytorch/vision', force_reload=True))
model = efficientnet_b7()

print(model)