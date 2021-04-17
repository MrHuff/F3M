from FFM_classes import *
import time



if __name__ == '__main__':
    n=1000000
    device = "cuda:0"
    X = torch.randn(n,3).float().to(device)
    min_points = float(2500)
    ref_points = 5000
    x_ref = X[0:ref_points,:]
    Y = torch.rand(n,3).float().to(device)
    b = torch.randn(n,1).float().to(device)
    ls = float(10.0)
    nr_of_interpolation = int(64)
    eff_var_limit=float(0.2)


    keops_benchmark_0 = benchmark_matmul(x_ref,X,ls=ls,device=device)
    keops_benchmark_1 = benchmark_matmul(x_ref,Y,ls=ls,device=device)

    true_0 = keops_benchmark_0@b
    true_1 = keops_benchmark_1@b


    FFM_X_0 = FFM(X=X,ls=ls,min_points=min_points,nr_of_interpolation=nr_of_interpolation,eff_var_limit=eff_var_limit,var_compression=False,smooth_interpolation=False,device=device)
    FFM_X_1 = FFM(X=X,ls=ls,min_points=min_points,nr_of_interpolation=nr_of_interpolation,eff_var_limit=eff_var_limit,var_compression=True,smooth_interpolation=False,device=device)
    FFM_X_2 = FFM(X=X,ls=ls,min_points=min_points,nr_of_interpolation=nr_of_interpolation,eff_var_limit=eff_var_limit,var_compression=True,smooth_interpolation=True,device=device)
    FFM_XY_1 = FFM(X=X,Y=Y,ls=ls,min_points=min_points,nr_of_interpolation=nr_of_interpolation,eff_var_limit=eff_var_limit,device=device)




    start = time.time()
    res_0 = FFM_X_0@b
    end= time.time()
    print(f'time exp 0: {end-start}')
    rel_err_0 = calc_rel_error(true_res=true_0,approx_res=res_0[:ref_points])
    print(f'err exp 0: {rel_err_0}')

    start = time.time()
    res_1 = FFM_X_1@b
    end= time.time()

    print(f'time exp 1: {end-start}')
    rel_err_1 = calc_rel_error(true_res=true_0,approx_res=res_1[:ref_points])
    print(f'err exp 1: {rel_err_1}')


    start = time.time()
    res_2 = FFM_X_2@b
    end= time.time()
    print(f'time exp 2: {end-start}')
    rel_err_2 = calc_rel_error(true_res=true_0,approx_res=res_2[:ref_points])
    print(f'err exp 2: {rel_err_2}')


    start = time.time()
    res_3 = FFM_XY_1@b
    end= time.time()

    print(f'time exp 3: {end-start}')
    rel_err_3 = calc_rel_error(true_res=true_1,approx_res=res_3[:ref_points])
    print(f'err exp 3: {rel_err_3}')



