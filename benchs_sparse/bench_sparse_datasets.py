import torch
from sparse_datasets import PlotData, MaternClusterData, FBMData
from FFMbench import FFMbench, PlotBench
import matplotlib.pyplot as plt
import numpy as np

import random
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

def ClusteredDataset2D():
    title = "ClusteredDataset2D"
    X = MaternClusterData(2,100,10000,.01)
    return X, title

def FBMDataset2D():
    title = "FBMDataset2D"
    X = FBMData(2,1000000,0.75)
    return X, title
    
todolist = [ClusteredDataset2D, FBMDataset2D]
    
for dataset_fun in todolist:
    X, title = dataset_fun()
    
    device = "cuda:0"                         # device
    
    X = X.float().to(device)
    n, dim = X.shape
    b = torch.randn(n,1).float().to(device)   # weights
    ls = float(1.0)                           # lengthscale
    
    bench_X = FFMbench(X,None,b,ls,title)
    
    
    
    # FFM parameters
    
    # we can try :
    # eff_var_limit = 0 to get more accuracy
    # or increase nb of interpolation points
    # or increase min_points
    
    # nr of interpolation nodes
    nr_of_interpolation = [int(64), int(128), int(256)]
           
    # Effective variance threshold
    eff_var_limit = [float(0.25), float(0.3), float(0.35), float(0.4)]            
    
    # stop when dividing when the largest box has 1000 points
    min_points = [float(250), float(500), float(1000), float(2000)]              
    
    var_compression = [True]
    
    smooth_interpolation = [False]
    
    
    
    res = bench_X(nr_of_interpolation = nr_of_interpolation,
            eff_var_limit = eff_var_limit,
            min_points = min_points,
            var_compression = var_compression, 
            smooth_interpolation = smooth_interpolation)
    
    import pickle
    
    f = open("benchs_sparse/results/"+title+".pkl", "wb")
    pickle.dump(res, f)
    f.close()
    
