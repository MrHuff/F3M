from F3M_src.FFM_classes import *
import time

if __name__ == '__main__':
    n=1000000 #Nr of observations
    device = "cuda:0" #device
    dim=3 #dims, stick to <5
    X = torch.rand(n,dim).float().to(device) #generate some data
    min_points = float(2500) # stop when dividing when the largest box has 1000 points
    ref_points = 5000 #calculate error on 5000 points
    x_ref = X[0:ref_points,:] #reference X
    b = torch.randn(n,1).float().to(device) #weights
    ls = float(5.0) #lengthscale
    sigma = torch.tensor(ls**0.5).float().to(device)
    nr_of_interpolation = int(64) #nr of interpolation nodes
    eff_var_limit=float(0.1) # Effective variance threshold
    small_field_limit = nr_of_interpolation
    FFM_obj = FFM(X=X,ls=ls,min_points=min_points,nr_of_interpolation=nr_of_interpolation,eff_var_limit=eff_var_limit,var_compression=True,smooth_interpolation=False,device=device,small_field_points=small_field_limit)
    K = ("gaussian",dim,1,sigma)
    sol = KernelLinearSolver(FFM_obj=FFM_obj,binding='torch',K=K,x=X,b=b,alpha=1,precond=True)

