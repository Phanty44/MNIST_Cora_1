import sklearn
import torch
import torchvision
import tabulate
from torch_geometric.datasets import Planetoid
from torch_geometric.loader import DataLoader

cora = Planetoid(".", "Cora")
data = cora[0]
print(data.x)
print(len(data.x[0]))
print(data.y)
print(len(data.y))