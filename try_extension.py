import torch
from torch.utils.cpp_extension import load
FFM = load(name='ffm_3d_float', sources=['n_tree.cu'])
if __name__ == '__main__':
    n=1000000
    device = "cuda:0"
    X = torch.randn(n,3).float().to(device)
    Y = torch.randn(n,3).float().to(device)
    b = torch.randn(n,1).float().to(device)
    ls = float(10.0)
    min_points = float(1000)
    nr_of_interpolation = int(64)
    var_compression = True
    eff_var_limit=float(0.1)
    smooth_flag = False

    y1 =  FFM.FFM_X_FLOAT_3(X,b,device,ls,min_points,nr_of_interpolation,var_compression,eff_var_limit)
    y2 =  FFM.SUPERSMOOTH_FFM_X_FLOAT_3(X,b,device,ls,min_points,nr_of_interpolation,eff_var_limit)
    print(y1[0:10])
    print(y2[0:10])

