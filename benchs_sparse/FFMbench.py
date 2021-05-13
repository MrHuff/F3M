   
import numpy as np
import time
from pykeops.torch import LazyTensor

def dict2str(d):
    return "\n".join(list(f"{k} : {v}" for k, v in d.items()))

def FFMbench(X, Y, b, sqls, title, Niter):
    
    import os, sys
    source_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.append(source_dir)
    os.chdir(source_dir)

    from FFM_classes import FFM
    from run_obj import benchmark_matmul, calc_rel_error_norm
    import itertools
    import torch
    
    
    # compute reference for error evaluation
    ref_points = 1000                         # calculate error on 1000 points
    x_ref = X[0:ref_points,:]                 # reference X
    device = X.device
    XY = X if Y is None else Y
        
    # compute reference using KeOps with double precision:
    #x_ref_keops = LazyTensor(x_ref[:,None,:].type(torch.cuda.DoubleTensor))
    #XY_keops = LazyTensor(XY[None,:,:].type(torch.cuda.DoubleTensor))
    #res_ref = ((-((x_ref_keops-XY_keops)**2).sum(dim=2)/2/sqls).exp() @ b.type(torch.cuda.DoubleTensor))
    
    # compute reference using KeOps with Kahan scheme:
    x_ref_keops = LazyTensor(x_ref[:,None,:])
    XY_keops = LazyTensor(XY[None,:,:])
    res_ref = (-((x_ref_keops-XY_keops)**2).sum(dim=2)/2/sqls).exp().__matmul__(b, sum_scheme="kahan_scheme")
    
    # compute reference using PyTorch :
    #x_ref_double = x_ref[:,None,:].type(torch.cuda.DoubleTensor)
    #XY_double = XY[None,:,:].type(torch.cuda.DoubleTensor)
    #res_ref = ((-((x_ref_double-XY_double)**2).sum(dim=2)/2/sqls).exp() @ b.type(torch.cuda.DoubleTensor))
    
    # compute reference using benchmark_matmul :
    #res_ref = benchmark_matmul(x_ref,XY,ls=sqls,device=device) @ b


    def call(**kwargs_list):
        elapsed = np.array([])
        rel_err = np.array([])
        kwargs_rec = []
        for k,params in enumerate(itertools.product(*kwargs_list.values())):
            kwargs = dict()
            for key, p in zip(kwargs_list,params):
                kwargs[key] = p
            kwargs_rec.append(kwargs)
            print("\ntesting with parameters :", dict2str(kwargs))
            myFFM = FFM(X=X, Y=Y, ls=sqls, device=device, **kwargs)
            elapsed_k = rel_err_k = 0
            for it in range(Niter):
                start = time.time()
                res = myFFM @ b
                end = time.time()
                elapsed_k += end-start
                rel_err_k += calc_rel_error_norm(true_res=res_ref, approx_res=res[:ref_points]).item()
            elapsed_k /= Niter
            rel_err_k /= Niter
            print(f'elapsed: {elapsed_k} s (averaged over {Niter} iteration(s))')
            print(f'error: {rel_err_k} (averaged over {Niter} iteration(s))')
            elapsed = np.append(elapsed, elapsed_k)
            rel_err = np.append(rel_err, rel_err_k)
        Xcpu, Ycpu, bcpu = X.cpu(), (Y.cpu() if Y is not None else None), b.cpu()
        return dict(X=Xcpu, Y=Ycpu, b=bcpu, sqls=sqls, elapsed=elapsed, rel_err=rel_err, kwargs_list=kwargs_list, kwargs_rec=kwargs_rec, title=title)
    
    return call

def PlotBench(dict_res):
    import matplotlib.pyplot as plt
    from hover_scatter import hover_scatter
    from sparse_datasets import PlotData
    elapsed = dict_res["elapsed"]
    rel_err = dict_res["rel_err"]
    kwargs_rec = dict_res["kwargs_rec"]
    kwargs_list = dict_res["kwargs_list"]
    shape_kwargs = tuple(len(kwarg) for kwarg in kwargs_list.values())[:-1]
    elapsed = elapsed.reshape(shape_kwargs)
    rel_err = rel_err.reshape(shape_kwargs)
    #plt.plot(elapsed[None,:], rel_err[None,:],'.')
    #plt.legend(kwargs_rec)
    names = list(dict2str(kwargs) for kwargs in kwargs_rec)
    hover_scatter(elapsed, rel_err,names, plotfun=plt.semilogy)
    plt.xlabel('time(s)')
    plt.ylabel('relative error')
    #plt.axis('equal')
    plt.title("benchmark results for "+dict_res["title"])
