import torch
x = torch.tensor([1, 2, 3, 4])
y=x.unsqueeze(-1)
print(y)