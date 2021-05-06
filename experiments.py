import pandas as pd
import numpy as np
from FFM_classes import *
import torch
from run_obj import *
import os
import os
columns = ['seed','n','d','effective_variance','min_points','small field limit','nr of node points','effective variance limit','relative error','time (s)']
import time

def experiment_1(device="cuda:0"):
    """
    k(X,X) - uniform distribution with varying effective variance, N and d.
    :return:
    """
    dirname = 'experiment_1'
    ref_points = 5000
    ls = float(1.0) #lengthscale
    counter = 0
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    for d in [3]:
        if d==1:
            node_list = [4,8,16]
        else:
            node_list = [4,5,6]
        for node_nr in node_list:
            nr_of_interpolation = int(node_nr**d)  # nr of interpolation nodes
            for evarlimit in [0.1,0.2,0.3,0.4,0.5]:
                eff_var_limit = float(evarlimit)
                for n, min_points, small_field_limit in zip([1000000, 10000000, 100000000, 1000000000],
                                                            [1000, 1000, 5000, 5000], [nr_of_interpolation, nr_of_interpolation, nr_of_interpolation, 2500]):
                    for r2 in [0.1,1,10,100]:
                        for seed in [1, 2, 3]:
                            if not os.path.exists(f'{dirname}/{dirname}_{counter}.csv'):
                                torch.manual_seed(seed)
                                X = torch.empty(n,d).uniform_(0, (r2*12)**0.5).to(device)
                                b = torch.empty(n, 1).normal_(0,1).float().to(device)  # weights
                                x_ref = X[0:ref_points, :]  # reference X
                                keops_benchmark_0 = benchmark_matmul(x_ref, X, ls=ls, device=device)  # get some references
                                FFM_obj= FFM(X=X, ls=ls, min_points=min_points, nr_of_interpolation=nr_of_interpolation,
                                              eff_var_limit=eff_var_limit, var_compression=True, smooth_interpolation=False,
                                              device=device, small_field_points=small_field_limit)
                                true_0 = keops_benchmark_0 @ b  # calculate reference
                                start = time.time()
                                res_0 = FFM_obj @ b
                                end = time.time()
                                calc_time = end-start
                                rel_err_0 = calc_rel_error(true_res=true_0, approx_res=res_0[:ref_points])
                                df = pd.DataFrame([[seed,n,d,r2,min_points,small_field_limit,nr_of_interpolation,eff_var_limit,rel_err_0.item(),calc_time]],columns=columns)
                                df.to_csv(f'{dirname}/{dirname}_{counter}.csv')
                                print('Wrote experiments: ',counter)
                                del X,b,x_ref,keops_benchmark_0,FFM_obj,true_0,res_0
                                torch.cuda.empty_cache()
                            counter+=1
                            print('counter: ',counter)


def experiment_2(device="cuda:0"):
    """
    k(X,X) - normal distribution with varying effective variance, N and d.
    :return:
    """
    recorded_data = []
    ref_points = 5000
    ls = float(1.0)  # lengthscale
    counter = 0
    dirname = 'experiment_2'
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    for d in [3]:
        if d == 1:
            node_list = [4, 8, 16]
        else:
            node_list = [4, 5, 6]
        for node_nr in node_list:
            nr_of_interpolation = int(node_nr ** d)  # nr of interpolation nodes
            for n, min_points, small_field_limit in zip([1000000, 10000000, 100000000, 1000000000],
                                                        [1000, 1000, 5000, 5000],
                                                        [nr_of_interpolation, nr_of_interpolation, nr_of_interpolation,
                                                         2500]):
                for evarlimit in [0.1,0.2,0.3,0.4,0.5]:
                    eff_var_limit = float(evarlimit)
                    for r2 in [0.1,1,10,100]:
                        for seed in [1, 2, 3]:
                            if not os.path.exists(f'{dirname}/{dirname}_{counter}.csv'):
                                torch.manual_seed(seed)
                                X = torch.empty(n, d).normal_(0, r2**0.5).to(device)
                                b = torch.empty(n, 1).normal_(0,1).float().to(device)  # weights
                                x_ref = X[0:ref_points, :]  # reference X
                                keops_benchmark_0 = benchmark_matmul(x_ref, X, ls=ls, device=device)  # get some references
                                FFM_obj = FFM(X=X, ls=ls, min_points=min_points, nr_of_interpolation=nr_of_interpolation,
                                              eff_var_limit=eff_var_limit, var_compression=True, smooth_interpolation=False,
                                              device=device, small_field_points=small_field_limit)
                                true_0 = keops_benchmark_0 @ b  # calculate reference
                                start = time.time()
                                res_0 = FFM_obj @ b
                                end = time.time()
                                calc_time = end-start
                                rel_err_0 = calc_rel_error(true_res=true_0, approx_res=res_0[:ref_points])
                                df = pd.DataFrame([[seed,n,d,r2,min_points,small_field_limit,nr_of_interpolation,eff_var_limit,rel_err_0.item(),calc_time]],columns=columns)
                                df.to_csv(f'{dirname}/{dirname}_{counter}.csv')
                                del X,b,x_ref,keops_benchmark_0,FFM_obj,true_0,res_0
                                torch.cuda.empty_cache()
                            counter += 1
                            print('counter: ',counter)


def experiment_3(device="cuda:0"):
    """
    k(X,Y) - uniform X, normal Y with varying effective variance, N and d. 0 distance between X and Y
    :return:
    """
    recorded_data = []
    ref_points = 5000
    ls = float(1.0)  # lengthscale
    counter = 0
    dirname = 'experiment_3'
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    for d in [3]:
        if d == 1:
            node_list = [4, 8, 16]
        else:
            node_list = [4, 5, 6]
        for node_nr in node_list:
            nr_of_interpolation = int(node_nr ** d)  # nr of interpo
            for n, min_points, small_field_limit in zip([1000000, 10000000, 100000000, 500000000],
                                                        [1000, 1000, 5000, 5000],
                                                        [nr_of_interpolation, nr_of_interpolation, nr_of_interpolation,
                                                         2500]):
                for evarlimit in [0.1,0.2,0.3,0.4,0.5]:
                    eff_var_limit = float(evarlimit)
                    for r2 in [0.1,1,10,100]:
                        for seed in [1, 2, 3]:
                            if not os.path.exists(f'{dirname}/{dirname}_{counter}.csv'):
                                torch.manual_seed(seed)
                                X = torch.empty(n,d).uniform_(0, (r2*12)**0.5).to(device)
                                Y = torch.empty(n, d).normal_(0, r2 ** 0.5).to(device)
                                b = torch.empty(n, 1).normal_(0,1).float().to(device)  # weights
                                x_ref = X[0:ref_points, :]  # reference X
                                keops_benchmark_0 = benchmark_matmul(x_ref, Y, ls=ls, device=device)  # get some references
                                FFM_obj = FFM(X=X,Y=Y, ls=ls, min_points=min_points, nr_of_interpolation=nr_of_interpolation,
                                              eff_var_limit=eff_var_limit, var_compression=True, smooth_interpolation=False,
                                              device=device, small_field_points=small_field_limit)
                                true_0 = keops_benchmark_0 @ b  # calculate reference
                                start = time.time()
                                res_0 = FFM_obj @ b
                                end = time.time()
                                calc_time = end-start
                                rel_err_0 = calc_rel_error(true_res=true_0, approx_res=res_0[:ref_points])
                                df = pd.DataFrame([[seed,n,d,r2,min_points,small_field_limit,nr_of_interpolation,eff_var_limit,rel_err_0.item(),calc_time]],columns=columns)
                                df.to_csv(f'{dirname}/{dirname}_{counter}.csv')
                                del X,Y,b,x_ref,keops_benchmark_0,FFM_obj,true_0,res_0
                                torch.cuda.empty_cache()
                            counter += 1
                            print('counter: ',counter)

def experiment_4(device="cuda:0"):
    """
    k(X,Y) - uniform X, normal Y with varying effective variance, N and d. Uniform has higher variance than Y
    :return:
    """
    recorded_data = []
    ref_points = 5000
    ls = float(1.0)  # lengthscale
    counter = 0
    dirname = 'experiment_3'
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    for d in [3]:
        if d == 1:
            node_list = [4, 8, 16]
        else:
            node_list = [4, 5, 6]
        for node_nr in node_list:
            nr_of_interpolation = int(node_nr ** d)  #
            for n, min_points, small_field_limit in zip([1000000, 10000000, 100000000, 500000000],
                                                        [1000, 1000, 5000, 5000],
                                                        [nr_of_interpolation, nr_of_interpolation, nr_of_interpolation,
                                                         2500]):
                for evarlimit in [0.1,0.2,0.3,0.4,0.5]:
                    eff_var_limit = float(evarlimit)
                    for r2 in [0.1,1,10,100]:
                        for seed in [1, 2, 3]:
                            if not os.path.exists(f'{dirname}/{dirname}_{counter}.csv'):
                                torch.manual_seed(seed)
                                X = torch.empty(n,d).uniform_(0, (r2*12)**0.5).to(device)
                                Y = torch.empty(n, d).normal_(0, 2*(r2**0.5)).to(device)
                                b = torch.empty(n, 1).normal_(0,1).float().to(device)  # weights
                                x_ref = X[0:ref_points, :]  # reference X
                                keops_benchmark_0 = benchmark_matmul(x_ref, Y, ls=ls, device=device)  # get some references
                                FFM_obj = FFM(X=X, Y=Y, ls=ls, min_points=min_points, nr_of_interpolation=nr_of_interpolation,
                                              eff_var_limit=eff_var_limit, var_compression=True, smooth_interpolation=False,
                                              device=device, small_field_points=small_field_limit)
                                true_0 = keops_benchmark_0 @ b  # calculate reference
                                start = time.time()
                                res_0 = FFM_obj @ b
                                end = time.time()
                                calc_time = end-start
                                rel_err_0 = calc_rel_error(true_res=true_0, approx_res=res_0[:ref_points])
                                df = pd.DataFrame([[seed,n,d,r2,min_points,small_field_limit,nr_of_interpolation,eff_var_limit,rel_err_0.item(),calc_time]],columns=columns)
                                df.to_csv(f'{dirname}/{dirname}_{counter}.csv')
                                del X,Y,b,x_ref,keops_benchmark_0,FFM_obj,true_0,res_0
                                torch.cuda.empty_cache()
                            counter += 1
                            print('counter: ',counter)


def experiment_5(device="cuda:0"):
    """
    k(X,Y) - uniform X, normal Y with varying effective variance, N and d. fixed distance from each other
    :return:
    """
    recorded_data = []
    ref_points = 5000
    ls = float(1.0)  # lengthscale
    counter = 0
    dirname = 'experiment_3'
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    for d in [3]:
        if d == 1:
            node_list = [4, 8, 16]
        else:
            node_list = [4, 5, 6]
        for node_nr in node_list:
            nr_of_interpolation = int(node_nr ** d)  #
            for n, min_points, small_field_limit in zip([1000000, 10000000, 100000000, 500000000],
                                                        [1000, 1000, 5000, 5000],
                                                        [nr_of_interpolation, nr_of_interpolation, nr_of_interpolation,
                                                         2500]):
                for evarlimit in [0.1,0.2,0.3,0.4,0.5]:
                    eff_var_limit = float(evarlimit)
                    for r2 in [0.1,1,10,100]:
                        for seed in [1, 2, 3]:
                            if not os.path.exists(f'{dirname}/{dirname}_{counter}.csv'):
                                torch.manual_seed(seed)
                                X = torch.empty(n,d).uniform_(0, (r2*12)**0.5).to(device)
                                Y = torch.empty(n, d).normal_(0, (r2 ** 0.5)).to(device)+2*r2
                                b = torch.empty(n, 1).normal_(0,1).float().to(device)  # weights
                                x_ref = X[0:ref_points, :]  # reference X
                                keops_benchmark_0 = benchmark_matmul(x_ref, Y, ls=ls, device=device)  # get some references
                                FFM_obj = FFM(X=X, Y=Y, ls=ls, min_points=min_points, nr_of_interpolation=nr_of_interpolation,
                                              eff_var_limit=eff_var_limit, var_compression=True, smooth_interpolation=False,
                                              device=device, small_field_points=small_field_limit)
                                true_0 = keops_benchmark_0 @ b  # calculate reference
                                start = time.time()
                                res_0 = FFM_obj @ b
                                end = time.time()
                                calc_time = end-start
                                rel_err_0 = calc_rel_error(true_res=true_0, approx_res=res_0[:ref_points])
                                df = pd.DataFrame([[seed,n,d,r2,min_points,small_field_limit,nr_of_interpolation,eff_var_limit,rel_err_0.item(),calc_time]],columns=columns)
                                df.to_csv(f'{dirname}/{dirname}_{counter}.csv')
                                del X,Y,b,x_ref,keops_benchmark_0,FFM_obj,true_0,res_0
                                torch.cuda.empty_cache()

                            counter += 1
                            print('counter: ',counter)

if __name__ == '__main__':
    experiment_1()
    experiment_2()
    experiment_3()
    experiment_4()
    experiment_5()