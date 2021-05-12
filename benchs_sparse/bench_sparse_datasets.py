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
    X = MaternClusterData(3,100,10000,.01)
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
    
todolist = [BMDataset2D]
    
for dataset_fun in todolist:
    X, title = dataset_fun()

    device = "cuda:0"                         # device
    X = X.float().to(device)
    n, dim = X.shape
    b = torch.randn(n,1).float().to(device)   # weights
    sqls = float(.01)**2                     # square of lengthscale
    
    bench_X = FFMbench(X,None,b,sqls,title)
    
    
    
    # FFM parameters
    
    # we can try :
    # eff_var_limit = 0 to get more accuracy
    # or increase nb of interpolation points
    # or increase min_points
    
    # nr of interpolation nodes
    nr_of_interpolation = [int(16), int(32), int(64), int(128), int(256), int(512)]

    # Effective variance threshold
    eff_var_limit = [float(0.1), float(0.3), float(0.35), float(0.4), float(0.45)]

    # stop when dividing when the largest box has 1000 points
    min_points = [float(250), float(500), float(1000), float(2000), float(4000)]              
    
    var_compression = [True]
    
    smooth_interpolation = [False]
    

    
    
    res = bench_X(nr_of_interpolation = nr_of_interpolation,
            eff_var_limit = eff_var_limit,
            min_points = min_points,
            var_compression = var_compression, 
            smooth_interpolation = smooth_interpolation)
            
    print("best time : ", min(res["elapsed"]))
    print("best accuracy : ", min(res["rel_err"]))
    
    import pickle
    
    f = open("benchs_sparse/results/"+title+".pkl", "wb")
    pickle.dump(res, f)
    f.close()
    
