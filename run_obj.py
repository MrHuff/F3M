from FFM_classes import *
import pandas as pd
import time
import numpy as np
import os
from tqdm import tqdm

def get_data_sampled(N, d, dist_1, dist_1_a, dist_1_b):
    if dist_1 is None:
        return None
    if dist_1 == 'normal':
        dat = torch.randn(N, d) * dist_1_b + dist_1_a
        return dat
    elif dist_1 == 'uniform':
        dist = torch.distributions.uniform.Uniform(dist_1_a, dist_1_b)
        dat = dist.sample(torch.Size([N,d]))
        return dat


def run_and_record_simulated(args_in):

    device = "cuda:0"
    save_path = args_in['save_path']
    job_index = args_in['job_index']
    N = int(args_in['N'])
    d = int(args_in['d'])
    ls = float(args_in['ls'])
    dist_1 =  args_in['dist_1']
    dist_1_a = args_in['dist_1_a']
    dist_1_b = args_in['dist_1_b']
    min_points = float(args_in['min_points'])
    nr_of_interpolation = int(args_in['nr_of_interpolation'])
    smooth = args_in['smooth']
    var_comp = args_in['var_comp']
    var_limit = float(args_in['var_limit'])
    ref_points = args_in['ref_points']
    n_loops = args_in['n_loops']
    dist_2  = args_in['dist_2']
    dist_2_a = args_in['dist_2_a']
    dist_2_b = args_in['dist_2_b']
    rows = []

    for i in tqdm(range(n_loops)):

        torch.manual_seed(i)
        X = get_data_sampled(N=N,d=d,dist_1=dist_1,dist_1_a=dist_1_a,dist_1_b=dist_1_b)
        X = X.float().to(device)
        x_ref = X[0:ref_points,:]
        Y = get_data_sampled(N=N, d=d, dist_1=dist_2, dist_1_a=dist_2_a, dist_1_b=dist_2_b)
        b = torch.randn(N,1)
        b = b.float().to(device)

        if Y is not None:
            Y = Y.float().to(device)
            smooth=False
            var_limit=None
            args_in['smooth'] = False
            args_in['var_limit']=None
            benchmark = benchmark_matmul(x_ref,Y,ls=ls,device=device)
            FFM_obj = FFM(X=X, Y=Y, ls=ls, min_points=min_points, nr_of_interpolation=nr_of_interpolation,
                           eff_var_limit=var_limit, device=device)
        else:
            dist_2=None
            dist_2_a=None
            dist_2_b=None
            args_in['dist_2'] = None
            args_in['dist_2_a'] = None
            if smooth:
                var_comp=False
            benchmark = benchmark_matmul(x_ref,X,ls=ls,device=device)
            FFM_obj = FFM(X=X, ls=ls, min_points=min_points, nr_of_interpolation=nr_of_interpolation,
                          eff_var_limit=var_limit, var_compression=var_comp, smooth_interpolation=smooth, device=device)

        true = benchmark@b
        start = time.time()
        res = FFM_obj @ b
        end = time.time()
        rel_err = calc_rel_error(true_res=true, approx_res=res[:ref_points])
        time_seconds = end-start
        rows.append([time_seconds,rel_err.item()])

    res_arr = np.array(rows)
    means = res_arr.mean(0)
    std = res_arr.std(0)
    cols = [key for key in args_in.keys()]
    cols.append('time_mean')
    cols.append('time_std')
    cols.append('error_mean')
    cols.append('error_std')
    vals = [val for val in args_in.values()]
    vals.append(means[0])
    vals.append(std[0])
    vals.append(means[1])
    vals.append(std[1])
    df = pd.DataFrame([vals],columns=cols)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    df.to_csv(save_path+f'/job_{job_index}.csv')

def calc_rel_error(true_res,approx_res):
    bool = torch.abs(true_res)>1e-6
    return torch.mean(torch.abs((true_res[bool].squeeze()-approx_res[bool].squeeze())/true_res[bool].squeeze()))

def calc_max_rel_error(true_res,approx_res):
    bool = torch.abs(true_res)>1e-6
    return torch.max(torch.abs((true_res[bool].squeeze()-approx_res[bool].squeeze())/true_res[bool].squeeze()))

def calc_abs_error(true_res,approx_res):
    return torch.mean(torch.abs((true_res.squeeze()-approx_res.squeeze())))

def calc_max_abs_error(true_res,approx_res):
    return torch.max(torch.abs((true_res.squeeze()-approx_res.squeeze())))