import pandas as pd
import numpy as np
import torch
from run_obj import *
import os
columns = ['eff_var','n','d','ls','effective_variance','min_points','small field limit','nr of node points','effective variance limit','R^2','time_CG','time_full','time_inference','penalty']
import time
import falkon
from conjugate_gradient.custom_falkon import custom_Falkon
from conjugate_gradient.custom_gaussian_kernel import custom_GaussianKernel
import argparse
import  pykeops

# pykeops.clean_pykeops()
# pykeops.test_torch_bindings()
from sklearn.metrics import roc_auc_score

def auc(true,pred):
    true=true.cpu().numpy()
    pred=pred.cpu().numpy()
    true_zero_one = np.clip(true,0,1)
    pred_zero_one = pred>0
    auc = roc_auc_score(true_zero_one,pred_zero_one)
    return auc

def job_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--idx', type=int, nargs='?', default=-1, help='which dataset to run')
    parser.add_argument('--seed', type=int, nargs='?', default=-1, help='which dataset to run')
    parser.add_argument('--penalty', type=float, nargs='?', default=-1, help='which dataset to run')
    return parser
def calc_R2(true,pred):
    var = true.var()
    mse = torch.mean((true-pred)**2)
    r2 = 1-(mse/var)
    return r2.item()



def calculate_results(seed,n,d,ls,effective_variance,min_points,small_field,nr_of_node_points,eff_var_limit,R_2,time_GC,time_full,time_inference,penalty):
    df = pd.DataFrame(
        [[seed,n,d,ls,effective_variance,min_points,small_field,nr_of_node_points,eff_var_limit,R_2,time_GC,time_full,time_inference,penalty]], columns=columns)
    return df

def generate_random_problem(X,prob_size,ls):
    perm = torch.randperm(prob_size)
    x_ref = X[perm]
    true_sol = torch.randn(prob_size,1)

    tmp = benchmark_matmul(X=X,Y=x_ref,ls=ls)

    solve_for = tmp@true_sol

    del tmp
    torch.cuda.empty_cache()
    return solve_for.cpu()


"""
FALKON vs FALKON roids

Best case, worst case data on 3d, real data, "worst case" vs best case
"""

def dataset_X(penalty_in, seed):
    dirname = f'FALKON_dataset_retry'
    M=100000
    N=1000000000
    d=2
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    nr_of_interpolation = 16
    penalty = penalty_in
    eff_var_limit = 2.0
    if not os.path.exists(f'{dirname}/{dirname}_lambda={penalty_in}_seed={seed}.csv'):
        torch.cuda.synchronize()
        problem_set = torch.load(f'real_problem_N={N}_seed={seed}.pt')
        X = problem_set['X']
        Y = problem_set['y']
        ls = problem_set['ls']
        kernel = custom_GaussianKernel(sigma=ls, min_points=250,
                                       var_compression=True,
                                       interpolation_nr=nr_of_interpolation,eff_var_limit=eff_var_limit)

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
                               nr_of_interpolation,eff_var_limit, r2, CG_TIME, TOTAL_TIME,
                               INFERENCE_TIME,penalty)
        df.to_csv(f'{dirname}/{dirname}_lambda={penalty_in}_seed={seed}.csv')
        del kernel, preds, model
        torch.cuda.empty_cache()
def dataset_X_bench(penalty_in, seed):

    # pykeops.clean_pykeops()
    # pykeops.test_torch_bindings()
    dirname = f'FALKON_dataset_benchmark'
    counter = 0
    N=1000000000
    M=100000
    d=2
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    nr_of_interpolation = 16
    penalty = penalty_in
    if not os.path.exists(f'{dirname}/{dirname}_lambda={penalty_in}_seed={seed}_bm.csv'):
        problem_set = torch.load(f'real_problem_N={N}_seed={seed}.pt')
        X = problem_set['X']
        Y = problem_set['y']
        ls = problem_set['ls']
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
                               INFERENCE_TIME,penalty)
        df.to_csv(f'{dirname}/{dirname}_lambda={penalty_in}_seed={seed}_bm.csv')
        print('Wrote experiments: ', counter)

        del kernel, preds, model
        torch.cuda.empty_cache()
    counter += 1
    print('counter: ', counter)



def dataset_taxi_F3M(dirname,penalty_in, seed,bench):
    M = 100000
    d = 3
    dirname = f'{dirname}_bench={bench}'
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    N=1000000000
    nr_of_interpolation = 64
    penalty = penalty_in
    eff_var_limit = 2.0
    if not os.path.exists(f'{dirname}/{dirname}_lambda={penalty_in}_seed={seed}.csv'):
        torch.cuda.synchronize()
        problem_set = torch.load('krr_taxi.pt')
        X = problem_set['X'][:N,:].contiguous()
        Y = problem_set['y'][:N,:].contiguous()
        ls = problem_set['ls']
        if bench:
            kernel = falkon.kernels.GaussianKernel(sigma=ls)
            options = falkon.FalkonOptions(use_cpu=False, debug=True, keops_memory_slack=0.25)
            model = custom_Falkon(kernel=kernel, penalty=penalty, M=M, options=options)

        else:
            kernel = custom_GaussianKernel(sigma=ls, min_points=1000,
                                           var_compression=True,
                                           interpolation_nr=nr_of_interpolation, eff_var_limit=eff_var_limit)

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

        # print(preds[:1000])
        # print(Y[:1000])
        r2 = auc(Y, preds)

        df = calculate_results(seed, X.shape[0], d, ls ** 2, 1 / ls ** 2, nr_of_interpolation, nr_of_interpolation,
                               nr_of_interpolation, eff_var_limit, r2, CG_TIME, TOTAL_TIME,
                               INFERENCE_TIME, penalty)
        print(df)
        df.to_csv(f'{dirname}/{dirname}_lambda={penalty_in}_seed={seed}.csv')
        del kernel, preds, model
        torch.cuda.empty_cache()


if __name__ == '__main__':
    input_args = vars(job_parser().parse_args())
    idx = input_args['idx']
    seed = input_args['seed']
    penalty_in = input_args['penalty']
    if idx==0:
        for seed,penalty_in in zip([1,2,3],[0.0075,0.01,0.0075]):
            dataset_X(penalty_in,seed)
    elif idx==1:
        for seed,penalty_in in zip([1,2,3],[1e-3,0.025,1e-3]):
            dataset_X_bench(penalty_in,seed)
    elif idx==2:
        dataset_taxi_F3M('taxi_krr_class_2_optimized_chunks',penalty_in,seed,False)
    elif idx==3:
        dataset_taxi_F3M('taxi_krr_class_2_9',penalty_in,seed,True)