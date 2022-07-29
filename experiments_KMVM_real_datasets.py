import pandas as pd
import numpy as np
from FFM_classes import *
import torch
from run_obj import *
import os
import pickle
columns = ['eff_var','n','d','effective_variance','min_points','small field limit','nr of node points','effective variance limit','relative error','relative error 2',
           'relative error max','abs error','abs error max','mean true','std true','min true','max true','time (s)']
import time
import argparse
import os
import copy
def job_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, nargs='?', default=-1, help='which dataset to run')
    parser.add_argument('--ablation', type=int, nargs='?',  default=0, help='which dataset to run')
    parser.add_argument('--chunk', type=int, nargs='?', default=0, help='which dataset to run')
    return parser
def calculate_results(true_0,res_0,ref_points,seed,n,d,r2,min_points,small_field_limit,nr_of_interpolation, eff_var_limit,calc_time):
    res_ref = res_0[:ref_points]
    mask = res_ref.isnan().any(dim=1)
    res_ref = res_ref[~mask]
    true_0 = true_0[~mask]
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



def experiment_real(dataset,ablation,chunk_idx,device="cuda:0"):
    """
    k(X,X) - uniform distribution with varying effective variance, N and d.
    :return:
    """
    if dataset == 'osm':
        X_base = torch.load('standardized_data_osm.pt')
    elif dataset == 'taxi':
        dat = torch.load('krr_taxi.pt')
        X_base = dat['X']
    _,d = X_base.shape
    dirname = f'{dataset}_ablation={ablation}'
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    ls = float(1.0 / 2 ** 0.5)
    if ablation==False:
        with open('real_kmvm_jobs_ablation.pkl', 'rb') as f:
            job_list_full = pickle.load(f)
    elif ablation==True:
        with open('real_kmvm_jobs.pkl', 'rb') as f:
            job_list_full = pickle.load(f)
    elif ablation==25:
        with open('real_kmvm_jobs.pkl', 'rb') as f:
            job_list_full = pickle.load(f)

    jobs_len = len(job_list_full)
    per_chunk = jobs_len//8
    chunked_list = [job_list_full[i:i + per_chunk] for i in range(0, len(job_list_full), per_chunk)]
    job_list = chunked_list[chunk_idx]
    ref_points = 5000
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    for job in job_list:
        s = job['seed']
        min_points = job['min_points']
        r2 = job['r2']
        eff_var_limit = job['eff_var_limit']
        nr_of_interpolation = int(job['node_nr']**d)
        small_field_limit = nr_of_interpolation
        n = job['n']
        counter = job['counter']
        if not os.path.exists(f'{dirname}/{dirname}_{counter}.csv'):
            torch.manual_seed(s)
            X = copy.deepcopy(X_base[:n,:]).contiguous() * r2
            # X = torch.empty(n, d).uniform_(0, (r2 * 12) ** 0.5)
            b = torch.empty(X.shape[0], 1).normal_(0, 1).contiguous()
            x_ref = X[:ref_points, :]  # reference X
            keops_benchmark_0 = benchmark_matmul(x_ref,X, ls=ls,
                                                        device=device)  # get some references
            true_0 = keops_benchmark_0 @ b  # calculate reference
            torch.cuda.synchronize()
            del keops_benchmark_0, x_ref
            torch.cuda.empty_cache()
            print("benchmarks done\n")
            torch.cuda.synchronize()
            if ablation==True:
                small_field_limit=int(0)
                FFM_obj = FFM(X=X, ls=ls, min_points=min_points, nr_of_interpolation=nr_of_interpolation,
                              eff_var_limit=eff_var_limit, var_compression=False,
                              device=device, small_field_points=small_field_limit)
            elif ablation==25:
                FFM_obj= FFM(X=X, ls=ls, min_points=min_points, nr_of_interpolation=nr_of_interpolation,
                              eff_var_limit=eff_var_limit, var_compression=False,
                              device=device, small_field_points=small_field_limit)
            elif ablation ==False:
                FFM_obj= FFM(X=X, ls=ls, min_points=min_points, nr_of_interpolation=nr_of_interpolation,
                             eff_var_limit=eff_var_limit, var_compression=True,
                             device=device, small_field_points=small_field_limit)
            torch.cuda.synchronize()
            start = time.time()
            res_0 = FFM_obj @ b
            end = time.time()
            torch.cuda.synchronize()
            calc_time = end-start
            df = calculate_results(true_0,res_0,ref_points,s,X.shape[0],d,r2,min_points,small_field_limit,nr_of_interpolation, eff_var_limit,calc_time)
            df.to_csv(f'{dirname}/{dirname}_{counter}.csv')
            print('Wrote experiments: ',counter)
            del FFM_obj,res_0
            torch.cuda.empty_cache()
            print('counter: ',counter)
            del X,b,true_0
            torch.cuda.empty_cache()


if __name__ == '__main__':
    input_args = vars(job_parser().parse_args())
    ds = input_args['dataset']
    ablation = bool(input_args['ablation'])
    chunk = input_args['chunk']
    experiment_real(ds,ablation,chunk)
