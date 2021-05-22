import torch
from sparse_datasets import PlotData, MaternClusterData, FBMData
from FFMbench import FFMbench, PlotBench
import matplotlib.pyplot as plt
import numpy as np

#import random
#random.seed(0)
#np.random.seed(0)
#torch.manual_seed(0)

class Container(torch.nn.Module):
    def __init__(self, my_values):
        super().__init__()
        for key in my_values:
            setattr(self, key, my_values[key])

def Uniform2D():
    title = "Uniform2D"
    X = torch.rand(1000000,2)
    return X, title

def Uniform3D():
    title = "Uniform3D"
    X = torch.rand(1000000,3)
    return X, title

def ClusteredDataset2D():
    title = "ClusteredDataset2D"
    X = MaternClusterData(2,100,10000,.01)
    return X, title

def ClusteredDataset3D():
    title = "ClusteredDataset3D"
    X = MaternClusterData(3,100,100000,.01)
    return X, title

def FBMDataset2D():
    title = "FBMDataset2D"
    X = FBMData(2,10000000,0.75)
    X = X - X.min(dim=0, keepdims=True)[0]
    X = X / X.max(dim=0, keepdims=True)[0]
    return X, title
    
def FBMDataset3D():
    title = "FBMDataset3D"
    X = FBMData(3,100000000,0.75)
    X = X - X.min(dim=0, keepdims=True)[0]
    X = X / X.max(dim=0, keepdims=True)[0]
    return X, title
    
def BMDataset2D():
    title = "BMDataset2D"
    X = FBMData(2,1000000,0.5)
    return X, title
    
def BMDataset3D():
    title = "BMDataset3D"
    X = FBMData(3,100000000,0.5)
    return X, title
    
todolist = [ClusteredDataset3D]
    
for dataset_fun in todolist:
    X, title = dataset_fun()
    X = X.float()
    X = X[0:5000000,:]
    print(X.shape)
    my_values = {
        'X':X.float(),
    }
    container = torch.jit.script(Container(my_values))
    container.save("../faulty_data.pt")
