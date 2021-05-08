from FFM_classes import *
import time
from run_obj import *
if __name__ == '__main__':
    n=1000000 #Nr of observations
    device = 'cuda:0' #device
    d=3 #dims, stick to <5
    r2=1
    min_points = float(2500)  # stop when dividing when the largest box has 1000 points
    ref_points = 5000  # calculate error on 5000 points
    X = torch.empty(n, d).uniform_(0, (r2 * 12) ** 0.5).to(device)
    Y = torch.empty(2500, d).normal_(0, r2 ** 0.5).to(device)
    b = torch.empty(2500, 1).normal_(0, 1).float().to(device)  # weights
    x_ref = X[0:ref_points, :]  # reference X
    torch.cuda.synchronize()
    ls = float(1.0) #lengthscale
    # ls = float(10.0) #lengthscale
    nr_of_interpolation = int(64) #nr of interpolation nodes
    eff_var_limit=float(0.1) # Effective variance threshold
    small_field_limit = int(64)
    # keops_benchmark_0 = benchmark_matmul(x_ref,X,ls=ls,device=device) #get some references
    keops_benchmark_1 = benchmark_matmul(x_ref,Y,ls=ls,device=device) #get some references
    # true_0 = keops_benchmark_0@b #calculate reference
    true_1 = keops_benchmark_1@b #calculate reference

    #Initialize different FFM objects
    # FFM_X_0 = FFM(X=X,ls=ls,min_points=min_points,nr_of_interpolation=nr_of_interpolation,eff_var_limit=eff_var_limit,var_compression=False,smooth_interpolation=True,device=device,small_field_points=small_field_limit)
    FFM_X_1 = FFM(X=X,Y=Y,ls=ls,min_points=min_points,nr_of_interpolation=nr_of_interpolation,eff_var_limit=eff_var_limit,var_compression=True,smooth_interpolation=False,device=device,small_field_points=small_field_limit)

    #Time computations
    # start = time.time()
    # res_0 = FFM_X_0@b
    # end= time.time()
    # print(f'time exp 0: {end-start}')
    # rel_err_0 = calc_rel_error(true_res=true_0,approx_res=res_0[:ref_points])
    # print(f'err exp 0: {rel_err_0}')

    start = time.time()
    res_1 = FFM_X_1@b
    end= time.time()
    print(f'time exp 1: {end-start}')
    rel_err_0 = calc_rel_error(true_res=true_1,approx_res=res_1[:ref_points])
    print(f'err exp 1: {rel_err_0}')
    print(true_1[:10])
    print(res_1[:10])
    print(res_1.shape)

