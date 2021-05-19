import pandas as pd
import numpy as np
from FFM_classes import *
import torch
from run_obj import *
import os
columns = ['seed','n','d','ls','effective_variance','min_points','small field limit','nr of node points','effective variance limit','R^2','time_CG','time_full','time_inference']
import time
import falkon
from conjugate_gradient.custom_falkon import custom_Falkon
from conjugate_gradient.custom_gaussian_kernel import custom_GaussianKernel
import argparse
import  pykeops
# pykeops.clean_pykeops()
# pykeops.test_torch_bindings()
def job_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--idx', type=int, nargs='?', default=-1, help='which dataset to run')
    return parser
def calc_R2(true,pred):
    var = true.var()
    mse = torch.mean((true-pred)**2)
    r2 = 1-(mse/var)
    return r2.item()
def calculate_results(seed,n,d,ls,effective_variance,min_points,small_field,nr_of_node_points,eff_var_limit,R_2,time_GC,time_full,time_inference):
    df = pd.DataFrame(
        [[seed,n,d,ls,effective_variance,min_points,small_field,nr_of_node_points,eff_var_limit,R_2,time_GC,time_full,time_inference]], columns=columns)
    return df

def generate_random_problem(X,prob_size,ls):
    perm = torch.randperm(prob_size)
    x_ref = X[perm]
    true_sol = torch.randn(prob_size,1)

    tmp = benchmark_matmul(X=X,Y=x_ref,ls=ls**2)

    solve_for = tmp@true_sol

    del tmp
    torch.cuda.empty_cache()
    return solve_for.cpu()


"""
FALKON vs FALKON roids

Best case, worst case data on 3d, real data, "worst case" vs best case
"""

def uniform_X():
    """
       k(X,X) - uniform distribution with varying effective variance, N and d.
       :return:
       """
    dirname = 'FALKON_uniform'
    counter = 0
    penalty = 1e-3
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    for seed in [1, 2, 3]:
        torch.manual_seed(seed)
        for d in [3]:
            # for n,M in zip([1000000, 10000000, 100000000, 1000000000],[10000,10000,10000,100000]):
            for n,M in zip([1000000000],[100000]):
                X = torch.empty(n, d).uniform_(0, (1 * 12) ** 0.5)
                for ls in [10,1,0.1]:
                    ls = ls ** 0.5
                    Y = generate_random_problem(X, 1000, ls)

                    for nr_of_interpolation in [27, 40, 64, 100]:
                        if not os.path.exists(f'{dirname}/{dirname}_{counter}.csv'):
                            torch.cuda.synchronize()
                            kernel = custom_GaussianKernel(sigma=ls, min_points=nr_of_interpolation,
                                                           var_compression=True,
                                                           interpolation_nr=nr_of_interpolation)

                            options = falkon.FalkonOptions(use_cpu=False, debug=False)
                            model = custom_Falkon(kernel=kernel, penalty=penalty, M=M, options=options)

                            start = time.time()
                            model.fit(X, Y)
                            end = time.time()
                            TOTAL_TIME = end - start
                            CG_TIME = model.conjugate_gradient_time

                            start = time.time()
                            preds = model.predict(X)
                            end = time.time()
                            INFERENCE_TIME = end - start
                            r2 = calc_R2(Y, preds)

                            df = calculate_results(seed, n, d, ls**2, 1/(ls**2), nr_of_interpolation, nr_of_interpolation,
                                              nr_of_interpolation, 0.1, r2, CG_TIME, TOTAL_TIME,
                                              INFERENCE_TIME)
                            df.to_csv(f'{dirname}/{dirname}_{counter}.csv')
                            print('Wrote experiments: ', counter)

                            del kernel, preds,model
                            torch.cuda.empty_cache()
                        counter += 1
                        print('counter: ', counter)


def uniform_X_benchmarks():
    """
       k(X,X) - uniform distribution with varying effective variance, N and d.
       :return:
       """
    dirname = 'FALKON_uniform_benchmarks'
    counter = 0
    penalty = 1e-3
    nr_of_interpolation = 0
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    for seed in [1, 2, 3]:
        torch.manual_seed(seed)
        for d in [3]:
            # for n,M in zip([1000000, 10000000, 100000000, 1000000000],[10000,10000,10000,100000]):
            for n, M in zip([1000000000], [100000]):
                X = torch.empty(n, d).uniform_(0, (1 * 12) ** 0.5)
                for ls in [10,1,0.1]:
                    ls = ls ** 0.5
                    Y = generate_random_problem(X, 1000, ls)
                    if not os.path.exists(f'{dirname}/{dirname}_{counter}.csv'):
                        kernel = falkon.kernels.GaussianKernel(sigma=ls)
                        options = falkon.FalkonOptions(use_cpu=False, debug=True, keops_memory_slack=0.25)
                        model = custom_Falkon(kernel=kernel, penalty=penalty, M=M, options=options)

                        start = time.time()
                        model.fit(X, Y)
                        end = time.time()
                        TOTAL_TIME = end - start
                        CG_TIME = model.conjugate_gradient_time

                        start = time.time()
                        preds = model.predict(X)
                        end = time.time()
                        INFERENCE_TIME = end - start
                        r2 = calc_R2(Y, preds)

                        df = calculate_results(seed, n, d, ls ** 2, 1 / (ls ** 2), nr_of_interpolation,
                                               nr_of_interpolation,
                                               nr_of_interpolation, 0.1, r2, CG_TIME, TOTAL_TIME,
                                               INFERENCE_TIME)
                        df.to_csv(f'{dirname}/{dirname}_{counter}.csv')
                        print('Wrote experiments: ', counter)
                        del kernel, preds, model
                        torch.cuda.empty_cache()
                    counter += 1
                    print('counter: ', counter)
def normal_X():
    dirname = 'FALKON_normal'
    counter = 0
    penalty = 1e-3
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    for seed in [1, 2, 3]:
        torch.manual_seed(seed)
        for d in [3]:
            # for n,M in zip([1000000, 10000000, 100000000, 1000000000],[10000,10000,10000,100000]):
            for n, M in zip([1000000000], [100000]):
                X = torch.empty(n, d).normal_(0, 1)
                for ls in [10, 1, 0.1]:
                    ls = ls ** 0.5
                    Y = generate_random_problem(X, 1000, ls)

                    for nr_of_interpolation in [27, 40, 64, 100]:
                        if not os.path.exists(f'{dirname}/{dirname}_{counter}.csv'):
                            torch.cuda.synchronize()
                            kernel = custom_GaussianKernel(sigma=ls, min_points=nr_of_interpolation,
                                                           var_compression=True,
                                                           interpolation_nr=nr_of_interpolation)

                            options = falkon.FalkonOptions(use_cpu=False, debug=True)
                            model = custom_Falkon(kernel=kernel, penalty=penalty, M=M, options=options)

                            start = time.time()
                            model.fit(X, Y)
                            end = time.time()
                            TOTAL_TIME = end - start
                            CG_TIME = model.conjugate_gradient_time

                            start = time.time()
                            preds = model.predict(X)
                            end = time.time()
                            INFERENCE_TIME = end - start
                            r2 = calc_R2(Y, preds)

                            df = calculate_results(seed, n, d, ls**2, 1/(ls**2), nr_of_interpolation, nr_of_interpolation,
                                              nr_of_interpolation, 0.1, r2, CG_TIME, TOTAL_TIME,
                                              INFERENCE_TIME)
                            df.to_csv(f'{dirname}/{dirname}_{counter}.csv')
                            print('Wrote experiments: ', counter)
                            counter += 1
                            print('counter: ', counter)
                            del kernel, preds, model
                            torch.cuda.empty_cache()
                        counter += 1
                        print('counter: ', counter)
def normal_X_bench():
    """
       k(X,X) - uniform distribution with varying effective variance, N and d.
       :return:
       """
    dirname = 'FALKON_normal_benchmarks'
    counter = 0
    penalty = 1e-3
    nr_of_interpolation = 0
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    for seed in [1, 2, 3]:
        torch.manual_seed(seed)
        for d in [3]:
            # for n,M in zip([1000000, 10000000, 100000000, 1000000000],[10000,10000,10000,100000]):
            for n, M in zip([1000000000], [100000]):
                X = torch.empty(n, d).normal_(0, 1)
                for ls in [10, 1, 0.1]:
                    ls = ls ** 0.5
                    Y = generate_random_problem(X, 1000, ls)
                    if not os.path.exists(f'{dirname}/{dirname}_{counter}.csv'):
                        kernel = falkon.kernels.GaussianKernel(sigma=ls)
                        options = falkon.FalkonOptions(use_cpu=False, debug=True, keops_memory_slack=0.25)
                        model = custom_Falkon(kernel=kernel, penalty=penalty, M=M, options=options)

                        start = time.time()
                        model.fit(X, Y)
                        end = time.time()
                        TOTAL_TIME = end - start
                        CG_TIME = model.conjugate_gradient_time

                        start = time.time()
                        preds = model.predict(X)
                        end = time.time()
                        INFERENCE_TIME = end - start
                        r2 = calc_R2(Y, preds)

                        df = calculate_results(seed, n, d, ls ** 2, 1 / (ls ** 2), nr_of_interpolation,
                                               nr_of_interpolation,
                                               nr_of_interpolation, 0.1, r2, CG_TIME, TOTAL_TIME,
                                               INFERENCE_TIME)
                        df.to_csv(f'{dirname}/{dirname}_{counter}.csv')
                        print('Wrote experiments: ', counter)

                        del kernel, preds, model
                        torch.cuda.empty_cache()
                    counter += 1
                    print('counter: ', counter)



def dataset_X():
    dirname = 'FALKON_dataset'
    counter = 0
    penalty = 1e-3
    M=100000
    d=2
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    for seed in [1, 2, 3]:
        torch.manual_seed(seed)
        X = torch.load('standardized_data_osm.pt')
        for ls in [10, 1, 0.1]:
            ls = ls ** 0.5
            Y = generate_random_problem(X, 1000, ls)
            for nr_of_interpolation in [9,16,25,36,64,100]:
                if not os.path.exists(f'{dirname}/{dirname}_{counter}.csv'):
                    torch.cuda.synchronize()
                    kernel = custom_GaussianKernel(sigma=ls, min_points=nr_of_interpolation,
                                                   var_compression=True,
                                                   interpolation_nr=nr_of_interpolation)

                    options = falkon.FalkonOptions(use_cpu=False, debug=False)
                    model = custom_Falkon(kernel=kernel, penalty=penalty, M=M, options=options)

                    start = time.time()
                    model.fit(X, Y)
                    end = time.time()
                    TOTAL_TIME = end - start
                    CG_TIME = model.conjugate_gradient_time

                    start = time.time()
                    preds = model.predict(X)
                    end = time.time()
                    INFERENCE_TIME = end - start
                    r2 = calc_R2(Y, preds)

                    df = calculate_results(seed, X.shape[0], d, ls**2, 1 / ls**2, nr_of_interpolation, nr_of_interpolation,
                                           nr_of_interpolation, 0.1, r2, CG_TIME, TOTAL_TIME,
                                           INFERENCE_TIME)
                    df.to_csv(f'{dirname}/{dirname}_{counter}.csv')
                    print('Wrote experiments: ', counter)
                    counter += 1
                    print('counter: ', counter)
                    del kernel, preds, model
                    torch.cuda.empty_cache()
                counter += 1
                print('counter: ', counter)
def dataset_X_bench():
    dirname = 'FALKON_dataset_benchmarks'
    counter = 0
    penalty = 1e-3
    nr_of_interpolation = 0
    M=100000
    d=2
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    for seed in [1, 2, 3]:
        torch.manual_seed(seed)
        X = torch.load('standardized_data_osm.pt')
        for ls in [10, 1, 0.1]:
            ls = ls ** 0.5
            Y = generate_random_problem(X, 1000, ls)
            if not os.path.exists(f'{dirname}/{dirname}_{counter}.csv'):
                kernel = falkon.kernels.GaussianKernel(sigma=ls)
                options = falkon.FalkonOptions(use_cpu=False, debug=True, keops_memory_slack=0.25)
                model = custom_Falkon(kernel=kernel, penalty=penalty, M=M, options=options)

                start = time.time()
                model.fit(X, Y)
                end = time.time()
                TOTAL_TIME = end - start
                CG_TIME = model.conjugate_gradient_time

                start = time.time()
                preds = model.predict(X)
                end = time.time()
                INFERENCE_TIME = end - start
                r2 = calc_R2(Y, preds)

                df = calculate_results(seed, X.shape[0], d, ls ** 2, 1 / ls ** 2, nr_of_interpolation,
                                       nr_of_interpolation,
                                       nr_of_interpolation, 0.1, r2, CG_TIME, TOTAL_TIME,
                                       INFERENCE_TIME)
                df.to_csv(f'{dirname}/{dirname}_{counter}.csv')
                print('Wrote experiments: ', counter)

                del kernel, preds, model
                torch.cuda.empty_cache()
            
if __name__ == '__main__':
    input_args = vars(job_parser().parse_args())
    idx = input_args['idx']
    if idx==1:
        uniform_X()
    elif idx==2:
        uniform_X_benchmarks()
    elif idx==3:
        normal_X()
    elif idx==4:
        normal_X_bench()
    elif idx==5:
        dataset_X()
    elif idx==6:
        dataset_X_bench()
