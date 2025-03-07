import time
from run_obj import *
if __name__ == '__main__':
    n=1000000 #Nr of observations
    device = "cuda:0" #device
    dim=3 #dims, stick to <5
    X = torch.rand(n,dim).float().to(device) #generate some data
    min_points = float(1000) # stop when dividing when the largest box has 1000 points
    ref_points = 5000 #calculate error on 5000 points
    x_ref = X[0:ref_points,:] #reference X
    Y = torch.rand(n,dim).float().to(device) #Generate Y if you want to do X Y interactions
    b = torch.randn(n,1).float().to(device) #weights
    ls = float(1.0) #lengthscale
    nr_of_interpolation = int(64) #nr of interpolation nodes
    eff_var_limit=float(0.1) # Effective variance threshold


    keops_benchmark_0 = benchmark_matmul(x_ref,X,ls=ls,device=device) #get some references
    keops_benchmark_1 = benchmark_matmul(x_ref,Y,ls=ls,device=device) #get some references

    true_0 = keops_benchmark_0@b #calculate reference
    true_1 = keops_benchmark_1@b #calculate reference

    #Initialize different FFM objects
    FFM_X_0 = block_FFM_matmul(X=X,ls=ls,min_points=min_points,nr_of_interpolation=nr_of_interpolation,eff_var_limit=eff_var_limit,var_compression=False,smooth_interpolation=False,device=device,blocks_X=2)
    FFM_X_1 = block_FFM_matmul(X=X,ls=ls,min_points=min_points,nr_of_interpolation=nr_of_interpolation,eff_var_limit=eff_var_limit,var_compression=True,smooth_interpolation=False,device=device,blocks_X=2)
    FFM_X_2 = block_FFM_matmul(X=X,ls=ls,min_points=min_points,nr_of_interpolation=nr_of_interpolation,eff_var_limit=eff_var_limit,var_compression=True,smooth_interpolation=True,device=device,blocks_X=2)
    FFM_XY_1 = block_FFM_matmul(X=X,Y=Y,ls=ls,min_points=min_points,nr_of_interpolation=nr_of_interpolation,eff_var_limit=eff_var_limit,device=device,blocks_X=2,blocks_Y=2)

    #Time computations
    start = time.time()
    res_0 = FFM_X_0@b
    end= time.time()
    print(f'time exp 0: {end-start}')
    rel_err_0 = calc_rel_error(true_res=true_0.cpu(),approx_res=res_0[:ref_points])
    print(f'err exp 0: {rel_err_0}')

    start = time.time()
    res_1 = FFM_X_1@b
    end= time.time()

    print(f'time exp 1: {end-start}')
    rel_err_1 = calc_rel_error(true_res=true_0.cpu(),approx_res=res_1[:ref_points])
    print(f'err exp 1: {rel_err_1}')

    start = time.time()
    res_2 = FFM_X_2@b
    end= time.time()
    print(f'time exp 2: {end-start}')
    rel_err_2 = calc_rel_error(true_res=true_0.cpu(),approx_res=res_2[:ref_points])
    print(f'err exp 2: {rel_err_2}')

    start = time.time()
    res_3 = FFM_XY_1@b
    end= time.time()

    print(f'time exp 3: {end-start}')
    rel_err_3 = calc_rel_error(true_res=true_1.cpu(),approx_res=res_3[:ref_points])
    print(f'err exp 3: {rel_err_3}')