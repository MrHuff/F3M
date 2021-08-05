import torch
from sparse_datasets import *
from FFMbench import FFMbench, PlotBench
import matplotlib.pyplot as plt
import numpy as np
import pickle
import sys
import random
import pykeops
import os
# pykeops.clean_pykeops()
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

class Container(torch.nn.Module):
    def __init__(self, my_values):
        super().__init__()
        for key in my_values:
            setattr(self, key, my_values[key])


def DoBench(todolist):

    todo_N = ["1e6", "1e7", "1e8","1e9"]                         # number of points
    Niters = [1, 1,1,1]  # nb of trials for each N
    # todo_N = ["1e9"]                         # number of points
    # Niters = [1]  # nb of trials for each N
    todo_ls = [float(.01), float(.1), float(1),float(10)]          # lengthscales
    
    # FFM parameters
    # nr of interpolation nodes
    nr_of_interpolation =  [int(64), int(125),int(216)]
    # Effective variance threshold
    # stop when dividing when the largest box has 1000 points
    # variance compression
    var_compression = [False]
    # eff_var_limit = [float(0.1), float(0.3),  float(0.5)]
    eff_var_limit = [float(0.0)]
    min_points = [float(5000)]

    res_folder = 'res_ablation'
    for dataset_fun in todolist:
        for Nstr, Niter in zip(todo_N,Niters):
            for ls in todo_ls:
                elapsed = rel_err = 0
                elapsed_rec = rel_err_rec = []
                for it in range(Niter):
                    X, title = eval(dataset_fun+Nstr)()
                    title += "_ls" + str(ls)
                    sqls = (ls/2)**0.5                           # square of lengthscale
                    X = X.float()
                    n, dim = X.shape
                    b = torch.randn(n,1).float()   # weights

                    bench_X = FFMbench(X,None,b,sqls,title,Niter=1)
                    
                    res = bench_X(nr_of_interpolation = nr_of_interpolation,
                            eff_var_limit = eff_var_limit,
                            min_points = min_points,
                            var_compression = var_compression,
                            small_field_points = [0])
                            
                    elapsed_rec.append(res["elapsed"])
                    rel_err_rec.append(res["rel_err"])
                            
                    elapsed += res["elapsed"]
                    rel_err += res["rel_err"]
                    
                elapsed /= Niter
                rel_err /= Niter
                
                print("best time : ", min(elapsed))
                print("best accuracy : ", min(rel_err))
                
                res["elapsed"] = elapsed
                res["rel_err"] = rel_err
                
                res["elapsed_rec"] = elapsed_rec
                res["rel_err_rec"] = rel_err_rec
                if not os.path.exists(f"benchs_sparse/{res_folder}/"):
                    os.makedirs(f"benchs_sparse/{res_folder}/")
                f = open(f"benchs_sparse/{res_folder}/"+title+".pkl", "wb")
                pickle.dump(res, f)
                f.close()
                    
                
if __name__ == "__main__":
    DoBench(list(sys.argv[1:]))
