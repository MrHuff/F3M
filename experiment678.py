import pandas as pd
import numpy as np
from FFM_classes import *
import torch
from run_obj import *
import os
columns = ['eff_var','n','d','effective_variance','min_points','small field limit','nr of node points','effective variance limit','relative error','relative error 2',
           'relative error max','abs error','abs error max','mean true','std true','min true','max true','time (s)']
import time
import argparse
import os
import pickle
from generate_jobs import *
def job_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--idx', type=int, nargs='?', default=-1, help='which dataset to run')
    return parser
def calculate_results(true_0,res_0,ref_points,seed,n,d,r2,min_points,small_field_limit,nr_of_interpolation, eff_var_limit,calc_time):
    res_ref = res_0[:ref_points]
    rel_err_0 = calc_rel_error(true_res=true_0, approx_res=res_ref)
    rel_err_1 = calc_max_rel_error(true_res=true_0, approx_res=res_ref)
    rel_err_2 = calc_abs_error(true_res=true_0, approx_res=res_ref)
    rel_err_3 = calc_max_abs_error(true_res=true_0, approx_res=res_ref)
    rel_err_4 = calc_rel_error_2(true_res=true_0, approx_res=res_ref)

    mean = true_0.mean().item()
    std = true_0.std().item()
    min = true_0.abs().min().item()
    max = true_0.abs().max().item()

    df = pd.DataFrame(
        [[seed, n, d, r2, min_points, small_field_limit, nr_of_interpolation, eff_var_limit, rel_err_0.item(),rel_err_4.item(),
          rel_err_1.item(), rel_err_2.item(), rel_err_3.item(), mean, std, min, max
             , calc_time]], columns=columns)
    return df


def experiment_6(chunk_idx,device="cuda:0"):
    """
    k(X,X) - uniform distribution with varying effective variance, N and d.
    :return:
    """
    dirname = 'experiment_6_hailmary'
    ref_points = 5000
    ls = float(1.0)/2**0.5 #lengthscale
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    with open('uniform_jobs.pkl', 'rb') as f:
        job_list_full = pickle.load(f)

    jobs_len = len(job_list_full)
    per_chunk = jobs_len//8
    chunked_list  = [job_list_full[i:i + per_chunk] for i in range(0, len(job_list_full), per_chunk)]
    job_list = chunked_list[chunk_idx]
    for job in job_list:
        seed = job['eff_var']
        n = job['n']
        d = job['d']
        nr_of_interpolation = job['nr_of_interpolation']
        min_points = job['min_points']
        small_field_limit = job['small_field_limit']
        r2 = job['r2']
        eff_var_limit = job['eff_var_limit']
        counter = job['counter']
        if (not os.path.exists(f'{dirname}/{dirname}_{counter}.csv')):
            torch.manual_seed(seed)
            X = torch.empty(n, d).uniform_(0, (12*r2) ** 0.5)
            b = torch.empty(n, 1).normal_(0, 1)  # weights
            x_ref = X[0:ref_points, :]
            keops_benchmark_0 = benchmark_matmul_double(x_ref, X, ls=ls,
                                                        device=device)  # get some references
            true_0 = keops_benchmark_0 @ b  # calculate reference
            torch.cuda.synchronize()
            del keops_benchmark_0, x_ref
            torch.cuda.empty_cache()
            print("benchmarks done\n")
            FFM_obj= FFM(X=X, ls=ls, min_points=min_points, nr_of_interpolation=nr_of_interpolation,
                         eff_var_limit=eff_var_limit, var_compression=True,
                         device=device, small_field_points=small_field_limit)
            torch.cuda.synchronize()
            start = time.time()
            res_0 = FFM_obj @ b
            end = time.time()
            torch.cuda.synchronize()
            calc_time = end-start
            df = calculate_results(true_0,res_0,ref_points,seed,n,d,r2,min_points,small_field_limit,nr_of_interpolation, eff_var_limit,calc_time)
            df.to_csv(f'{dirname}/{dirname}_{counter}.csv')
            print('Wrote experiments: ',counter,'\n')
            del FFM_obj, res_0,X, b, true_0
            torch.cuda.empty_cache()

def experiment_7(chunk_idx,device="cuda:0"):
    """
    k(X,X) - uniform distribution with varying effective variance, N and d.
    :return:
    """
    dirname = 'experiment_7_hailmary'
    ref_points = 5000
    ls = float(1.0)/2**0.5 #lengthscale
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    with open('normal_jobs.pkl', 'rb') as f:
        job_list_full = pickle.load(f)
    jobs_len = len(job_list_full)
    per_chunk = jobs_len//8
    chunked_list  = [job_list_full[i:i + per_chunk] for i in range(0, len(job_list_full), per_chunk)]
    job_list = chunked_list[chunk_idx]
    for job in job_list:
        seed = job['eff_var']
        n = job['n']
        d = job['d']
        nr_of_interpolation = job['nr_of_interpolation']
        min_points = job['min_points']
        small_field_limit = job['small_field_limit']
        r2 = job['r2']
        eff_var_limit = job['eff_var_limit']
        counter = job['counter']
        print(job)
        if (not os.path.exists(f'{dirname}/{dirname}_{counter}.csv')):
            torch.manual_seed(seed)
            X = torch.empty(n, d).normal_(0, r2 ** 0.5)
            b = torch.empty(n, 1).normal_(0, 1)  # weights
            x_ref = X[0:ref_points, :]
            keops_benchmark_0 = benchmark_matmul_double(x_ref, X, ls=ls,
                                                        device=device)  # get some references
            true_0 = keops_benchmark_0 @ b  # calculate reference
            torch.cuda.synchronize()
            del keops_benchmark_0, x_ref
            torch.cuda.empty_cache()
            print("benchmarks done\n")
            FFM_obj= FFM(X=X, ls=ls, min_points=min_points, nr_of_interpolation=nr_of_interpolation,
                          eff_var_limit=eff_var_limit, var_compression=True,
                          device=device, small_field_points=small_field_limit)
            torch.cuda.synchronize()
            start = time.time()
            res_0 = FFM_obj @ b
            end = time.time()
            torch.cuda.synchronize()
            calc_time = end-start
            df = calculate_results(true_0,res_0,ref_points,seed,n,d,r2,min_points,small_field_limit,nr_of_interpolation, eff_var_limit,calc_time)
            df.to_csv(f'{dirname}/{dirname}_{counter}.csv')
            print('Wrote experiments: ',counter)
            del X, b, true_0,FFM_obj, res_0
            torch.cuda.empty_cache()

def experiment_8(chunk_idx,device="cuda:0"):
    """
    k(X,Y) - uniform X, normal Y with varying effective variance, N and d. 0 distance between X and Y
    :return:
    """
    ref_points = 5000
    ls = float(1.0)/2**0.5 #lengthscale
    dirname = 'experiment_8_hailmary'
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    with open('mix_jobs.pkl', 'rb') as f:
        job_list_full = pickle.load(f)
    jobs_len = len(job_list_full)
    per_chunk = jobs_len // 8
    chunked_list = [job_list_full[i:i + per_chunk] for i in range(0, len(job_list_full), per_chunk)]
    job_list = chunked_list[chunk_idx]
    for job in job_list:
        seed = job['eff_var']
        n = job['n']
        d = job['d']
        nr_of_interpolation = job['nr_of_interpolation']
        min_points = job['min_points']
        small_field_limit = job['small_field_limit']
        r2 = job['r2']
        eff_var_limit = job['eff_var_limit']
        counter = job['counter']
        print(job)
        if not os.path.exists(f'{dirname}/{dirname}_{counter}.csv'):
            torch.manual_seed(seed)
            X = torch.empty(n, d).uniform_(0, (r2 * 12) ** 0.5)
            Y = torch.empty(n, d).normal_(0, r2 ** 0.5)
            b = torch.empty(n, 1).normal_(0, 1)  # weights
            x_ref = X[0:ref_points, :]  # reference X
            keops_benchmark_0 = benchmark_matmul_double(x_ref, Y, ls=ls,
                                                        device=device)  # get some references
            true_0 = keops_benchmark_0 @ b  # calculate reference
            torch.cuda.synchronize()
            del keops_benchmark_0, x_ref
            torch.cuda.empty_cache()
            print("benchmarks done\n")
            torch.cuda.synchronize()
            FFM_obj = FFM(X=X,Y=Y, ls=ls, min_points=min_points, nr_of_interpolation=nr_of_interpolation,
                          eff_var_limit=eff_var_limit, var_compression=True,
                          device=device, small_field_points=small_field_limit)
            torch.cuda.synchronize()
            start = time.time()
            res_0 = FFM_obj @ b
            end = time.time()
            torch.cuda.synchronize()
            calc_time = end-start
            df = calculate_results(true_0, res_0, ref_points, seed, n, d, r2, min_points,
                                   small_field_limit, nr_of_interpolation, eff_var_limit,
                                   calc_time)
            df.to_csv(f'{dirname}/{dirname}_{counter}.csv')
            del X,Y, b, true_0,FFM_obj, res_0
            torch.cuda.empty_cache()

if __name__ == '__main__':
    generate_jobs_uniform()
    generate_jobs_normal()
    generate_jobs_mix()
    input_args = vars(job_parser().parse_args())
    idx = input_args['idx']
    if idx==7:
        experiment_7(chunk_idx=1)
    if idx == 6:
        experiment_6(chunk_idx=1)
    if idx == 8:
        experiment_8(chunk_idx=1)