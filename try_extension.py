from FFM_classes import *
import time
if __name__ == '__main__':
    n=1000000
    device = "cuda:0"
    X = torch.randn(n,3).float().to(device)
    min_points = float(5000)
    x_ref = X[0:int(min_points)]
    Y = torch.rand(n,3).float().to(device)
    b = torch.randn(n,1).float().to(device)
    ls = float(10.0)
    nr_of_interpolation = int(64)
    var_compression = True
    eff_var_limit=float(0.1)

    FFM_X_0 = FFM(X=X,ls=ls,min_points=min_points,nr_of_interpolation=nr_of_interpolation,eff_var_limit=eff_var_limit,var_compression=False,smooth_interpolation=False)
    FFM_X_1 = FFM(X=X,ls=ls,min_points=min_points,nr_of_interpolation=nr_of_interpolation,eff_var_limit=eff_var_limit,var_compression=True,smooth_interpolation=False)
    FFM_X_2 = FFM(X=X,ls=ls,min_points=min_points,nr_of_interpolation=nr_of_interpolation,eff_var_limit=eff_var_limit,var_compression=True,smooth_interpolation=True)
    FFM_XY_1 = FFM(X=X,Y=Y,ls=ls,min_points=min_points,nr_of_interpolation=nr_of_interpolation,eff_var_limit=eff_var_limit)

    start = time.time()
    res_0 = FFM_X_0@b
    end= time.time()
    print(f'time exp 0: {end-start}')

    start = time.time()
    res_1 = FFM_X_1@b
    end= time.time()

    print(f'time exp 1: {end-start}')


    start = time.time()
    res_2 = FFM_X_2@b
    end= time.time()
    print(f'time exp 2: {end-start}')


    start = time.time()
    res_3 = FFM_XY_1@b
    end= time.time()

    print(f'time exp 3: {end-start}')



