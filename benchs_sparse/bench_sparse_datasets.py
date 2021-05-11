import torch
from sparse_datasets import PlotData, MaternClusterData, FBMData
from FFMbench import FFMbench, PlotBench
import matplotlib.pyplot as plt
import numpy as np

import random
random.seed(0)
np.random.seed(0)
torch.manual_seed(1)

class Container(torch.nn.Module):
    def __init__(self, my_values):
        super().__init__()
        for key in my_values:
            setattr(self, key, my_values[key])

def ClusteredDataset2D():
    title = "ClusteredDataset2D"
    X = MaternClusterData(2,100,10000,.01)
    return X, title

def ClusteredDataset3D():
    title = "ClusteredDataset3D"
    X = MaternClusterData(3,100,10000,.01)
    return X, title

def FBMDataset2D():
    title = "FBMDataset2D"
    X = FBMData(2,1000000,0.75)
    return X, title
    
def FBMDataset3D():
    title = "FBMDataset3D"
    X = FBMData(3,1000000,0.75)
    return X, title
    
def BMDataset2D():
    title = "BMDataset2D"
    X = FBMData(2,1000000,0.25)
    return X, title
    
def BMDataset3D():
    title = "BMDataset3D"
    X = FBMData(3,1000000,0.5)
    return X, title
    
todolist = [ClusteredDataset2D, FBMDataset2D]
    
for dataset_fun in todolist:
    X, title = dataset_fun()

    device = "cuda:0"                         # device
    # X = X.float()
    # my_values = {
    #     'X':X.float(),
    # }
    # container = torch.jit.script(Container(my_values))
    # container.save("../faulty_data.pt")


    X = X.float().to(device)


    n, dim = X.shape
    b = torch.ones(n,1).float().to(device)   # weights
    # b = torch.randn(n,1).float().to(device)   # weights
    # b = torch.rand(n,1).float().to(device)*100   # weights
    ls = float(1.0)                           # lengthscale
    
    bench_X = FFMbench(X,None,b,ls,title)
    
    
    
    # FFM parameters
    
    # we can try :
    # eff_var_limit = 0 to get more accuracy
    # or increase nb of interpolation points
    # or increase min_points
    
    # nr of interpolation nodes
    # nr_of_interpolation = [int(64), int(128), int(256)]
    nr_of_interpolation = [int(16)]

    # Effective variance threshold
    eff_var_limit = [float(0.1), float(0.3), float(0.35), float(0.4)]

    # stop when dividing when the largest box has 1000 points
    min_points = [float(250), float(500), float(1000), float(2000)]              
    
    var_compression = [True, False]
    
    smooth_interpolation = [True, False]
    
    
    
    res = bench_X(nr_of_interpolation = nr_of_interpolation,
            eff_var_limit = eff_var_limit,
            min_points = min_points,
            var_compression = var_compression, 
            smooth_interpolation = smooth_interpolation)
    
    import pickle
    
    f = open("benchs_sparse/results/"+title+".pkl", "wb")
    pickle.dump(res, f)
    f.close()
    
