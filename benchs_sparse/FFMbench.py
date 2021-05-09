    
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir('..')

from FFM_classes import *
from run_obj import *
import itertools



def FFMbench(X, Y, b, ls):

    # compute reference with KeOps for error evaluation
    ref_points = 5000                         # calculate error on 5000 points
    x_ref = X[0:ref_points,:]                 # reference X
    device = X.device
    if Y is None:
        res_ref = benchmark_matmul(x_ref,X,ls=ls,device=device) @ b
    else:
        res_ref = benchmark_matmul(x_ref,Y,ls=ls,device=device) @ b

    def call(**kwargs_list):
        kwargs = kwargs_list
        print(kwargs_list)
        for params in itertools.product(*kwargs_list.values()):
            for key, p in zip(kwargs,params):
                kwargs[key] = p
            print("\ntesting with parameters :", kwargs)
            myFFM = FFM(X=X, Y=Y, ls=ls, device=device, **kwargs)
            start = time.time()
            res = myFFM @ b
            end = time.time()
            print(f'elapsed: {end-start}')
            rel_err = calc_rel_error(true_res=res_ref, approx_res=res[:ref_points])
            print(f'error: {rel_err}')
    
    return call
