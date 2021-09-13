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

def experiment_1(device="cuda:0"):
    """
    k(X,X) - uniform distribution with varying effective variance, N and d.
    :return:
    """
    dirname = 'experiment_1_78D_uniform'
    ref_points = 5000
    ls = float(1.0)/2**0.5 #lengthscale
    counter = 0
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    for seed in [1]:
        for d in [7,8]:
            nr_of_interpolation = 5000
            for n, min_points, small_field_limit in zip([1000000, 10000000, ],
                                                        [5000, 5000, ],
                                                        [nr_of_interpolation, nr_of_interpolation, nr_of_interpolation]):
                for r2 in [0.1, 1, 10]:
                    torch.manual_seed(seed)
                    X = torch.empty(n, d).uniform_(0, (r2 * 12) ** 0.5)
                    b = torch.empty(n, 1).normal_(0, 1)
                    x_ref = X[0:ref_points, :]  # reference X
                    keops_benchmark_0 = benchmark_matmul_double(x_ref, X, ls=ls, device=device)  # get some references
                    true_0 = keops_benchmark_0 @ b  # calculate reference
                    torch.cuda.synchronize()
                    del keops_benchmark_0,x_ref
                    torch.cuda.empty_cache()
                    print("benchmarks done\n")
                    for evarlimit in [0.1,0.3,0.5]:
                        eff_var_limit = float(evarlimit)*d
                        if not os.path.exists(f'{dirname}/{dirname}_{counter}.csv'):
                            torch.cuda.synchronize()
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
                            del FFM_obj,res_0
                            torch.cuda.empty_cache()
                        counter+=1
                        print('counter: ',counter)
                    del X,b,true_0
                    torch.cuda.empty_cache()

def experiment_2(device="cuda:0"):
    """
    k(X,X) - normal distribution with varying effective variance, N and d.
    :return:
    """
    recorded_data = []
    ref_points = 5000
    ls = float(1.0)/2**0.5 #lengthscale
    counter = 0
    dirname = 'experiment_2_78D_uniform'
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    for seed in [1]:
        for d in [7, 8]:
            nr_of_interpolation = 5000
            for n, min_points, small_field_limit in zip([1000000, 10000000, ],
                                                        [5000, 20000,],
                                                        [nr_of_interpolation, nr_of_interpolation,
                                                         nr_of_interpolation]):  # nr of interpolation nodes

                for r2 in [0.1, 1, 10]:
                    X = 0
                    b = 0
                    true_0 = 0
                    if not os.path.exists(f'{dirname}/{dirname}_{counter}.csv'):
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
                    for evarlimit in [0.1,0.3,0.5]:
                        eff_var_limit = float(evarlimit)*d
                        if not os.path.exists(f'{dirname}/{dirname}_{counter}.csv'):
                            torch.cuda.synchronize()

                            FFM_obj = FFM(X=X, ls=ls, min_points=min_points, nr_of_interpolation=nr_of_interpolation,
                                          eff_var_limit=eff_var_limit, var_compression=True,
                                          device=device, small_field_points=small_field_limit)
                            torch.cuda.synchronize()
                            start = time.time()
                            res_0 = FFM_obj @ b
                            end = time.time()
                            calc_time = end-start
                            torch.cuda.synchronize()

                            df = calculate_results(true_0, res_0, ref_points, seed, n, d, r2, min_points,
                                                   small_field_limit, nr_of_interpolation, eff_var_limit,
                                                   calc_time)

                            df.to_csv(f'{dirname}/{dirname}_{counter}.csv')
                            del FFM_obj,res_0
                            torch.cuda.empty_cache()
                        counter += 1
                        print('counter: ',counter)
                    del X,b,true_0
                    torch.cuda.empty_cache()

def experiment_3(device="cuda:0"):
    """
    k(X,Y) - uniform X, normal Y with varying effective variance, N and d. 0 distance between X and Y
    :return:
    """
    recorded_data = []
    ref_points = 5000
    ls = float(1.0)/2**0.5 #lengthscale
    counter = 0
    dirname = 'experiment_3_78D_uniform'
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    for seed in [1]:
        for d in [7, 8]:
            nr_of_interpolation = 5000
            for n, min_points, small_field_limit in zip([1000000, 10000000, ],
                                                        [5000, 5000, ],
                                                        [nr_of_interpolation, nr_of_interpolation,
                                                         nr_of_interpolation]):
                for r2 in [0.1, 1, 10]:
                    torch.manual_seed(seed)
                    X = torch.empty(n, d).uniform_(0, (r2 * 12) ** 0.5)
                    Y = torch.empty(n, d).normal_(0, r2 ** 0.5)
                    b = torch.empty(n, 1).normal_(0, 1)
                    x_ref = X[0:ref_points, :]  # reference X
                    keops_benchmark_0 = benchmark_matmul_double(x_ref, Y, ls=ls, device=device)  # get some references
                    true_0 = keops_benchmark_0 @ b  # calculate reference
                    torch.cuda.synchronize()
                    del keops_benchmark_0,x_ref
                    torch.cuda.empty_cache()
                    print("benchmarks done\n")
                    for evarlimit in [0.1,0.3,0.5]:
                        eff_var_limit = float(evarlimit)*d
                        if not os.path.exists(f'{dirname}/{dirname}_{counter}.csv'):
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
                            del FFM_obj, res_0
                            torch.cuda.empty_cache()
                        counter += 1
                        print('counter: ', counter,'\n')
                    del X, b, true_0
                    torch.cuda.empty_cache()


if __name__ == '__main__':
    input_args = vars(job_parser().parse_args())
    idx = input_args['idx']
    if idx==1:
        experiment_1()
    elif idx==2:
        experiment_2()
    elif idx==3:
        experiment_3()
