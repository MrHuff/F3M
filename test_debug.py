from FFM_classes import *
import time
from run_obj import *
if __name__ == '__main__':
    N = 1000000000
    d = 3
    ls = 3.0
    penalty = 1e-5
    M = 10000
    X = torch.rand(N, d)
    b = torch.randn(M, 1)
    Y = X[:M]
    min_points = 64
    f_obj = FFM(X=X,Y=Y,ls=ls,min_points=min_points,eff_var_limit=0.1,var_compression=True,small_field_points=64)
    start = time.time()
    f_obj@b
    end = time.time()
    print(end-start)
    del f_obj,X,Y,b
    torch.cuda.empty_cache()
    X = torch.randn(N, d)
    b = torch.randn(N, 1)
    Y = X[:M]

    f_obj = FFM(X=Y,Y=X,ls=ls,min_points=min_points,eff_var_limit=0.1,var_compression=True,small_field_points=64)
    start = time.time()
    f_obj@b
    end = time.time()
    print(end-start)





