import torch
from sparse_datasets import *
from FFMbench import FFMbench, PlotBench
import matplotlib.pyplot as plt
import numpy as np
import pickle
import sys

import random
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

class Container(torch.nn.Module):
    def __init__(self, my_values):
        super().__init__()
        for key in my_values:
            setattr(self, key, my_values[key])


def DoBench(todolist):
    
    todo_N = ["1e6","1e7","1e8"]                         # number of points 
    Niters = [100,10,1]  # nb of trials for each N
    todo_ls = [float(.01), float(.1), float(1)]          # lengthscales
    
    # FFM parameters
    # nr of interpolation nodes
    nr_of_interpolation = [int(16), int(32), int(64), int(128), int(256), int(512)]
    # Effective variance threshold
    eff_var_limit = [float(0.1), float(0.3), float(0.35), float(0.4), float(0.45)]
    # stop when dividing when the largest box has 1000 points
    min_points = [float(250), float(500), float(1000), float(2000), float(4000)]              
    # variance compression 
    var_compression = [True]
                
    for dataset_fun in todolist:
        for Nstr, Niter in zip(todo_N,Niters):
            for ls in todo_ls:
                elapsed = rel_err = 0
                for it in range(Niter):
                    X, title = eval(dataset_fun+Nstr)()
                    title += "_ls" + str(ls)
                
                    sqls = ls**2                              # square of lengthscale
                    device = "cuda:0"                         # device
                    X = X.float().to(device)
                    n, dim = X.shape
                    b = torch.randn(n,1).float().to(device)   # weights              
                    
                    bench_X = FFMbench(X,None,b,sqls,title,Niter=1)
                    
                    res = bench_X(nr_of_interpolation = nr_of_interpolation,
                            eff_var_limit = eff_var_limit,
                            min_points = min_points,
                            var_compression = var_compression)
                            
                    elapsed += res["elapsed"]
                    rel_err += res["rel_err"]
                    
                elapsed /= Niter
                rel_err /= Niter
                
                print("best time : ", min(elapsed))
                print("best accuracy : ", min(rel_err))
                
                res["elapsed"] = elapsed
                res["rel_err"] = rel_err
    
                f = open("benchs_sparse/results/"+title+".pkl", "wb")
                pickle.dump(res, f)
                f.close()
                    
                
if __name__ == "__main__":
    DoBench(list(sys.argv[1:]))
