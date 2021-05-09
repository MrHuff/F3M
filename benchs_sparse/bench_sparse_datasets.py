import torch
from sparse_datasets import PlotData, MaternClusterData, FBMData
from FFMbench import FFMbench

#X = MaternClusterData(2,100,100,.05)

X = FBMData(2,1000000,0.75)

device = "cuda:0"                         # device

X = X.float().to(device)
n, dim = X.shape
b = torch.randn(n,1).float().to(device)   # weights
ls = float(1.0)                           # lengthscale

bench_X = FFMbench(X,None,b,ls)



# FFM parameters

# we can try :
# eff_var_limit = 0 to get more accuracy
# or increase nb of interpolation points
# or increase min_points

# nr of interpolation nodes
nr_of_interpolation = [int(32), int(64), int(128)]
       
# Effective variance threshold
eff_var_limit = [float(0), float(0.1), float(0.15), float(0.2)]            

# stop when dividing when the largest box has 1000 points
min_points = [float(500), float(1000), float(2000)]              

var_compression = [True]

smooth_interpolation = [False]



bench_X(nr_of_interpolation = nr_of_interpolation,
        eff_var_limit = eff_var_limit,
        min_points = min_points,
        var_compression = var_compression, 
        smooth_interpolation = smooth_interpolation)
