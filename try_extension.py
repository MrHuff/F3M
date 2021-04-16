import torch
from torch.utils.cpp_extension import load
FFM = load(name='ffm_3d_float', sources=['pybinder_setup.cu'])
if __name__ == '__main__':
    n=1000000
    device = "cuda:0"
    X = torch.randn(n,3).float().to(device)
    min_points = float(5000)
    x_ref = X[0:int(min_points)]
    # Y = torch.randn(n,3).float().to(device)
    b = torch.randn(n,1).float().to(device)
    ls = float(5.0)
    nr_of_interpolation = int(30)
    var_compression = True
    eff_var_limit=float(0.1)
    y_ref = FFM.rbf_float_3(x_ref,X,b,ls,True)
    # y1 =  FFM.FFM_X_FLOAT_3(X,b,device,ls,min_points,nr_of_interpolation,var_compression,eff_var_limit)
    y1 =  FFM.SUPERSMOOTH_FFM_X_FLOAT_3(X,b,device,ls,min_points,nr_of_interpolation,eff_var_limit)
    print(y_ref[0:10])
    print(y1[0:10])
    # print(y2[0:10])

