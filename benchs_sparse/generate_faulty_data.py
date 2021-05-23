import torch
from sparse_datasets import *
from FFMbench import FFMbench, PlotBench
import matplotlib.pyplot as plt
import numpy as np

class Container(torch.nn.Module):
    def __init__(self, my_values):
        super().__init__()
        for key in my_values:
            setattr(self, key, my_values[key])

todolist = [ClusteredDataset3D1e7]

for dataset_fun in todolist:
    X, title = dataset_fun()
    X = X.float()
    print(X.shape)
    my_values = {
        'X':X.float(),
    }
    container = torch.jit.script(Container(my_values))
    container.save("../faulty_data.pt")
