from FFM_classes import *
import time
from run_obj import *
if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')# good solution !!!!
    N = 100000000
    d = 3
    ls = 3.0
    penalty = 1e-5
    M = 10000
    X = torch.rand(M, d)
    b = torch.randn(N, 1)
    Y = torch.randn(N,d)
    min_points = 64
    obj_test = par_FFM(X=X,Y=Y,ls=ls,min_points=min_points,eff_var_limit=0.1,var_compression=False,small_field_points=64,par_factor=2)
    # obj_test = FFM(X=X,Y=Y,ls=ls,min_points=min_points,eff_var_limit=0.1,var_compression=False,small_field_points=64)
    start = time.time()
    obj_test@b
    end = time.time()
    print(end-start)






