import pandas as pd
import numpy as np
from FFM_classes import *
import torch
from run_obj import *
import os
columns = ['seed','n','d','effective_variance','min_points','small field limit','nr of node points','effective variance limit','relative error','relative error 2',
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
    dirname = 'experiment_1_07'
    ref_points = 5000
    ls = float(1.0)/2**0.5 #lengthscale
    counter = 0
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    for seed in [1]:
        for d in [3]:
            node_list = [4,5,6]
            for node_nr in node_list:
                nr_of_interpolation = int(node_nr ** d)
                for n, min_points, small_field_limit in zip([1000000, 10000000, 100000000, 1000000000],
                                                            [1000, 1000, 5000, 5000],
                                                            [nr_of_interpolation, nr_of_interpolation, nr_of_interpolation,
                                                             nr_of_interpolation]):
                    for r2 in [0.1, 1, 10, 100]:
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
                        for evarlimit in [0.7]:
                            eff_var_limit = float(evarlimit)
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
    dirname = 'experiment_2_27'
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    for seed in [1]:
        for d in [3]:
            node_list = [3]
            for node_nr in node_list:
                nr_of_interpolation = int(node_nr ** d)  # nr of interpolation nodes
                for n, min_points, small_field_limit in zip([1000000, 10000000, 100000000, 1000000000],
                                                            [2500, 2500, 5000, 10000],
                                                            [nr_of_interpolation, nr_of_interpolation, nr_of_interpolation,
                                                             2500]):
                    for r2 in [0.1, 1, 10, 100]:
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
                        for evarlimit in [0.1, 0.3, 0.5]:
                            eff_var_limit = float(evarlimit)
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
    dirname = 'experiment_3'
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    for seed in [1]:
        for d in [3]:
            if d == 1:
                node_list = [4, 8, 16]
            else:
                node_list = [4, 5, 6]
            for node_nr in node_list:
                nr_of_interpolation = int(node_nr ** d)  # nr of interpo
                for n, min_points, small_field_limit in zip([1000000, 10000000, 100000000, 500000000],
                                                            [2500, 2500, 2500, 2500],
                                                            [nr_of_interpolation, nr_of_interpolation, nr_of_interpolation,
                                                             nr_of_interpolation]):
                    for r2 in [0.1, 1, 10, 100]:
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
                            eff_var_limit = float(evarlimit)
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

def experiment_4(device="cuda:0"):
    """
    k(X,Y) - uniform X, normal Y with varying effective variance, N and d. Uniform has higher variance than Y
    :return:
    """
    recorded_data = []
    ref_points = 5000
    ls = float(1.0)/2**0.5 #lengthscale
    counter = 0
    dirname = 'experiment_4'
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    for seed in [1]:
        for d in [3]:
            if d == 1:
                node_list = [4, 8, 16]
            else:
                node_list = [4, 5, 6]
            for node_nr in node_list:
                nr_of_interpolation = int(node_nr ** d)  #
                for n, min_points, small_field_limit in zip([1000000, 10000000, 100000000, 500000000],
                                                            [2500, 2500, 5000, 5000],
                                                            [nr_of_interpolation, nr_of_interpolation, nr_of_interpolation,
                                                             2500]):
                    for r2 in [0.1, 1, 10, 100]:
                        torch.manual_seed(seed)
                        X = torch.empty(n, d).uniform_(0, (r2 * 12) ** 0.5)
                        Y = torch.empty(n, d).normal_(0, 2 * (r2 ** 0.5))
                        b = torch.empty(n, 1).normal_(0, 1)  # weights
                        x_ref = X[0:ref_points, :]  # reference X
                        keops_benchmark_0 = benchmark_matmul_double(x_ref, Y, ls=ls, device=device)  # get some references
                        true_0 = keops_benchmark_0 @ b  # calculate reference
                        torch.cuda.synchronize()
                        del keops_benchmark_0,x_ref
                        torch.cuda.empty_cache()
                        print("benchmarks done\n")
                        for evarlimit in [0.1,0.3,0.5]:
                            eff_var_limit = float(evarlimit)
                            if not os.path.exists(f'{dirname}/{dirname}_{counter}.csv'):
                                torch.cuda.synchronize()
                                FFM_obj = FFM(X=X, Y=Y, ls=ls, min_points=min_points, nr_of_interpolation=nr_of_interpolation,
                                              eff_var_limit=eff_var_limit, var_compression=True, 
                                              device=device, small_field_points=small_field_limit)
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


def experiment_5(device="cuda:0"):
    """
    k(X,Y) - uniform X, normal Y with varying effective variance, N and d. fixed distance from each other
    :return:
    """
    recorded_data = []
    ref_points = 5000
    ls = float(1.0)/2**0.5 #lengthscale
    counter = 0
    dirname = 'experiment_5'
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    for seed in [1, 1000, 3]:
        for d in [3]:
            if d == 1:
                node_list = [4, 8, 16]
            else:
                node_list = [4, 5, 6]
            for node_nr in node_list:
                nr_of_interpolation = int(node_nr ** d)  #
                for n, min_points, small_field_limit in zip([1000000, 10000000, 100000000, 500000000],
                                                            [2500, 2500, 5000, 5000],
                                                            [nr_of_interpolation, nr_of_interpolation, nr_of_interpolation,
                                                             2500]):
                    for r2 in [0.1, 1, 10, 100]:
                        torch.manual_seed(seed)
                        X = torch.empty(n, d).uniform_(0, (r2 * 12) ** 0.5)
                        Y = torch.empty(n, d).normal_(0, (r2 ** 0.5)) + 2 * r2
                        b = torch.empty(n, 1).normal_(0, 1)  # weights
                        x_ref = X[0:ref_points, :]  # reference X
                        keops_benchmark_0 = benchmark_matmul_double(x_ref, Y, ls=ls, device=device)  # get some references
                        true_0 = keops_benchmark_0 @ b  # calculate reference
                        torch.cuda.synchronize()
                        del keops_benchmark_0,x_ref
                        torch.cuda.empty_cache()
                        print("benchmarks done\n")
                        for evarlimit in [0.1,0.3,0.5]:
                            eff_var_limit = float(evarlimit)
                            if (not os.path.exists(f'{dirname}/{dirname}_{counter}.csv')) and r2 < 100 and evarlimit>0.3:
                                FFM_obj = FFM(X=X, Y=Y, ls=ls, min_points=min_points, nr_of_interpolation=nr_of_interpolation,
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

def experiment_6(device="cuda:0"):
    """
    k(X,X) - uniform distribution with varying effective variance, N and d.
    :return:
    """
    dirname = 'experiment_6_256_rest'
    ref_points = 5000
    ls = float(1.0)/2**0.5 #lengthscale
    counter = 0
    # node_list = [1024, 512, 256]
    node_list = [256]
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    for seed in [1]:
        for d in [4,5]:
            for node_nr in node_list:
                nr_of_interpolation = node_nr  # nr of interpolation nodes

                for n, min_points, small_field_limit in zip([1000000, 10000000, 100000000, 500000000],
                                                            [2500, 2500, 10000, 20000], [nr_of_interpolation, nr_of_interpolation, nr_of_interpolation, nr_of_interpolation]):
                    for r2 in [0.1,1,10]:
                        X=0
                        b=0
                        true_0 = 0
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
                        for evarlimit in [0.1, 0.3, 0.5]:
                            eff_var_limit = float(evarlimit)
                            if (not os.path.exists(f'{dirname}/{dirname}_{counter}.csv')):
                                print(n,r2)
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
                                del FFM_obj, res_0
                                torch.cuda.empty_cache()
                            counter += 1
                            print('counter: ', counter,'\n')
                        del X, b, true_0
                        torch.cuda.empty_cache()

def experiment_7(device="cuda:0"):
    """
    k(X,X) - uniform distribution with varying effective variance, N and d.
    :return:
    """
    dirname = 'experiment_7_5'
    ref_points = 5000
    ls = float(1.0)/2**0.5 #lengthscale
    counter = 0
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    for seed in [0]:
        for d in [4,5]:
            node_list = [256, 512, 1024]
            for nr_of_interpolation in node_list:
                for n, min_points, small_field_limit in zip([1000000, 10000000, 100000000, 500000000],
                                                            [5000, 5000, 20000, 20000],
                                                            [nr_of_interpolation, 2500,
                                                             10000, 10000]):
                    if d>4 and n==500000000:
                        n=500000000/2
                    for r2 in [0.1,1,10]:
                        X=0
                        b=0
                        true_0 = 0
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
                        for evarlimit in [0.1, 0.3, 0.5]:
                            print(n)
                            eff_var_limit = float(evarlimit)
                            if (not os.path.exists(f'{dirname}/{dirname}_{counter}.csv')):
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
                                # except Exception as e:
                                #     df = calculate_results(torch.zeros(10,1),torch.zeros(10,1),ref_points,seed,n,d,r2,min_points,small_field_limit,nr_of_interpolation, eff_var_limit,np.inf)
                                #     df.to_csv(f'{dirname}/{dirname}_{counter}.csv')
                                #     print(e)
                                del FFM_obj, res_0
                                torch.cuda.empty_cache()
                            counter += 1
                            print('counter: ', counter,'\n')
                    del X, b, true_0
                    torch.cuda.empty_cache()
def experiment_8(device="cuda:0"):
    """
    k(X,Y) - uniform X, normal Y with varying effective variance, N and d. 0 distance between X and Y
    :return:
    """
    recorded_data = []
    ref_points = 5000
    ls = float(1.0)/2**0.5 #lengthscale
    counter = 0
    dirname = 'experiment_8'
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    node_list = [256,512,1024]
    for seed in [1]:
        for d in [4,5]:
            for node_nr in node_list:
                nr_of_interpolation =node_nr  # nr of interpo
                for n, min_points, small_field_limit in zip([1000000, 10000000, 100000000, 250000000],
                                                            [2500, 2500, 5000, 10000],
                                                            [nr_of_interpolation, nr_of_interpolation, nr_of_interpolation,
                                                             nr_of_interpolation]):
                    for r2 in [0.1,1,10]:
                        torch.manual_seed(seed)

                        X = torch.empty(n, d).uniform_(0, (r2 * 12) ** 0.5)
                        Y = torch.empty(n, d).normal_(0, r2 ** 0.5)
                        b = torch.empty(n, 1).normal_(0, 1)  # weights
                        x_ref = X[0:ref_points, :]  # reference X
                        keops_benchmark_0 = benchmark_matmul_double(x_ref, Y, ls=ls, device=device)  # get some references
                        true_0 = keops_benchmark_0 @ b  # calculate reference
                        torch.cuda.synchronize()
                        del keops_benchmark_0,x_ref
                        torch.cuda.empty_cache()
                        print("benchmarks done\n")
                        for evarlimit in [0.1, 0.3, 0.5]:
                            eff_var_limit = float(evarlimit)
                            # print(seed, n, d, r2, min_points,
                            #       small_field_limit, nr_of_interpolation, eff_var_limit, )
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


def experiment_9(device="cuda:0"):
    """
    k(X,X) - uniform distribution with varying effective variance, N and d.
    :return:
    """
    dirname = 'experiment_9'
    ref_points = 5000
    counter = 0
    seed=0
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    min_points = 5000
    for d in [2]:
        for ls in [100,10,1,0.1,1e-2]:
            ls=ls**0.5/2**0.5
            evarlimit_list= [0.1,0.3,0.5]
            for evarlimit in evarlimit_list:
                node_list = [3, 4, 5, 6, 7, 8, 9, 10]
                for node_nr in node_list:
                    nr_of_interpolation = int(node_nr ** d)
                    small_field_limit = nr_of_interpolation
                    eff_var_limit = float(evarlimit)
                    if not os.path.exists(f'{dirname}/{dirname}_{counter}.csv'):
                        torch.cuda.synchronize()
                        X = torch.load('standardized_data_osm.pt')
                        n = X.size(0)
                        b = torch.empty(n, 1).normal_(0, 1)
                        x_ref = X[0:ref_points, :]  # reference X
                        keops_benchmark_0 = benchmark_matmul_double(x_ref, X, ls=ls, device=device)  # get some references
                        true_0 = keops_benchmark_0 @ b  # calculate reference
                        torch.cuda.synchronize()
                        del keops_benchmark_0
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
                        df = calculate_results(true_0,res_0,ref_points,seed,n,d,1/ls,min_points,small_field_limit,nr_of_interpolation, eff_var_limit,calc_time)
                        df.to_csv(f'{dirname}/{dirname}_{counter}.csv')
                        print('Wrote experiments: ',counter)
                        del FFM_obj,res_0,X,b,x_ref,true_0
                        torch.cuda.empty_cache()
                    counter+=1
                    print('counter: ',counter)
            torch.cuda.empty_cache()

def experiment_10(device="cuda:0"):
    """
    k(X,X) - normal distribution with varying effective variance, N and d.
    :return:
    """
    recorded_data = []
    ref_points = 5000
    ls = float(1.0)/2**0.5 #lengthscale
    counter = 0
    dirname = 'experiment_10_3'
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    for seed in [3]:
        for d in [3]:
            node_list = [3,4,5]
            for node_nr in node_list:
                nr_of_interpolation = int(node_nr ** d)  # nr of interpolation nodes
                for n, min_points, small_field_limit in zip([1000000, 10000000, 100000000, 1000000000],
                                                            [2500, 2500, 5000, 5000],
                                                            [nr_of_interpolation, nr_of_interpolation, nr_of_interpolation,
                                                             nr_of_interpolation]):
                    for r2 in [1]:
                        X = 0
                        b = 0
                        true_0 = 0
                        if not os.path.exists(f'{dirname}/{dirname}_{counter}.csv'):
                            torch.manual_seed(seed)
                            X = torch.empty(n, d).normal_(0, r2 ** 0.5)
                            X = X/((X*X).sqrt().sum(1,keepdim=True))
                            b = torch.empty(n, 1).normal_(0, 1)  # weights
                            x_ref = X[0:ref_points, :]
                            keops_benchmark_0 = benchmark_matmul_double(x_ref, X, ls=ls,
                                                                        device=device)  # get some references
                            true_0 = keops_benchmark_0 @ b  # calculate reference
                            torch.cuda.synchronize()
                            del keops_benchmark_0, x_ref
                            torch.cuda.empty_cache()
                            print("benchmarks done\n")
                        for evarlimit in [0.1, 0.3, 0.5, 1.0]:
                            eff_var_limit = float(evarlimit)
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

if __name__ == '__main__':
    input_args = vars(job_parser().parse_args())
    idx = input_args['idx']
    if idx==1:
        experiment_1()
    elif idx==2:
        experiment_2()
    elif idx==3:
        experiment_3()
    elif idx==4:
        experiment_4()
    elif idx==5:
        experiment_5()
    elif idx==6:
        experiment_6()
    elif idx==7:
        experiment_7()
    elif idx==8:
        experiment_8()
    elif idx==9:
        experiment_9()
    elif idx==10:
        experiment_10()