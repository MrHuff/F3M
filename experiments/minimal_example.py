import time
from run_obj import *
if __name__ == '__main__':
    n=1000000 #Nr of observations
    device = "cuda:0" #device
    dim=3 #dims, stick to <5
    X = torch.randn(n,dim).float().to(device) #generate some data
    min_points = float(2500) # stop when dividing when the largest box has 1000 points
    ref_points = 5000 #calculate error on 5000 points
    x_ref = X[0:ref_points,:] #reference X
    Y = torch.rand(n,dim).float().to(device) #Generate Y if you want to do X Y interactions
    b = torch.randn(n,1).float().to(device) #weights
    Y_2 = torch.rand(n//2,dim).float().to(device)
    b_2 = torch.randn(n//2,1).float().to(device) #weights
    ls = float(1) #lengthscale
    nr_of_interpolation = int(64) #nr of interpolation nodes
    eff_var_limit=float(0.1) # Effective variance threshold
    small_field_limit = nr_of_interpolation


    keops_benchmark_0 = benchmark_matmul_double(x_ref,X,ls=ls,device=device) #get some references
    keops_benchmark_1 = benchmark_matmul_double(x_ref,Y,ls=ls,device=device) #get some references
    keops_benchmark_2 = benchmark_matmul_double(x_ref,Y_2,ls=ls,device=device) #get some references

    true_0 = keops_benchmark_0@b #calculate reference
    true_1 = keops_benchmark_1@b #calculate reference
    true_2 = keops_benchmark_2@b_2 #calculate reference

    #Initialize different FFM objects
    FFM_X_0 = FFM(X=X,ls=ls,min_points=min_points,nr_of_interpolation=nr_of_interpolation,eff_var_limit=eff_var_limit,var_compression=False, device=device,small_field_points=small_field_limit)
    FFM_X_1 = FFM(X=X,ls=ls,min_points=min_points,nr_of_interpolation=nr_of_interpolation,eff_var_limit=eff_var_limit,var_compression=True, device=device,small_field_points=small_field_limit)
    FFM_X_2 = FFM(X=X,ls=ls,min_points=min_points,nr_of_interpolation=nr_of_interpolation,eff_var_limit=eff_var_limit,var_compression=True,device=device,small_field_points=small_field_limit)
    FFM_XY_1 = FFM(X=X,Y=Y,ls=ls,min_points=min_points,nr_of_interpolation=nr_of_interpolation,eff_var_limit=eff_var_limit,device=device,small_field_points=small_field_limit,var_compression=True)
    FFM_XY_2 = FFM(X=X,Y=Y_2,ls=ls,min_points=min_points,nr_of_interpolation=nr_of_interpolation,eff_var_limit=eff_var_limit,device=device,small_field_points=small_field_limit,var_compression=True)

    #Time computations
    start = time.time()
    res_0 = FFM_X_0@b
    end= time.time()
    print(f'time exp 0: {end-start}')
    rel_err_0 = calc_rel_error_2(true_res=true_0,approx_res=res_0[:ref_points])
    print(f'err exp 0: {rel_err_0}')

    start = time.time()
    res_1 = FFM_X_1@b
    end= time.time()

    print(f'time exp 1: {end-start}')
    rel_err_1 = calc_rel_error_2(true_res=true_0,approx_res=res_1[:ref_points])
    print(f'err exp 1: {rel_err_1}')

    start = time.time()
    res_2 = FFM_X_2@b
    end= time.time()
    print(f'time exp 2: {end-start}')
    rel_err_2 = calc_rel_error_2(true_res=true_0,approx_res=res_2[:ref_points])
    print(f'err exp 2: {rel_err_2}')

    start = time.time()
    res_3 = FFM_XY_1@b
    end= time.time()

    print(f'time exp 3: {end-start}')
    rel_err_3 = calc_rel_error_2(true_res=true_1,approx_res=res_3[:ref_points])
    print(f'err exp 3: {rel_err_3}')

    start = time.time()
    res_4 = FFM_XY_2@b_2
    end= time.time()

    print(f'time exp 4: {end-start}')
    rel_err_4 = calc_rel_error_2(true_res=true_2,approx_res=res_4[:ref_points])
    print(f'err exp 4: {rel_err_4}')