   
import numpy as np
import time

def dict2str(d):
    return "\n".join(list(f"{k} : {v}" for k, v in d.items()))

def FFMbench(X, Y, b, ls, title):
    
    import os, sys
    source_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.append(source_dir)
    os.chdir(source_dir)

    from FFM_classes import FFM
    from run_obj import benchmark_matmul, calc_rel_error_norm
    import itertools
    
    # compute reference with KeOps for error evaluation
    ref_points = 5000                         # calculate error on 5000 points
    x_ref = X[0:ref_points,:]                 # reference X
    device = X.device
    if Y is None:
        res_ref = benchmark_matmul(x_ref,X,ls=ls,device=device) @ b
    else:
        res_ref = benchmark_matmul(x_ref,Y,ls=ls,device=device) @ b

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
            myFFM = FFM(X=X, Y=Y, ls=ls, device=device, **kwargs)
            start = time.time()
            res = myFFM @ b
            end = time.time()
            elapsed = np.append(elapsed, end-start)
            print(f'elapsed: {end-start}')
            rel_err = np.append(rel_err, calc_rel_error_norm(true_res=res_ref, approx_res=res[:ref_points]).item())
            print(f'error: {rel_err[-1]}')
            print(res_ref[:10])
            print(res[:10])
        Xcpu, Ycpu, bcpu = X.cpu(), (Y.cpu() if Y is not None else None), b.cpu()
        return dict(X=Xcpu, Y=Ycpu, b=bcpu, ls=ls, elapsed=elapsed, rel_err=rel_err, kwargs_rec=kwargs_rec, title=title)
    
    return call

def PlotBench(dict_res):
    import matplotlib.pyplot as plt
    from hover_scatter import hover_scatter
    from sparse_datasets import PlotData
    elapsed = dict_res["elapsed"]
    rel_err = dict_res["rel_err"]
    kwargs_rec = dict_res["kwargs_rec"]
    #plt.plot(elapsed[None,:], rel_err[None,:],'.')
    #plt.legend(kwargs_rec)
    names = list(dict2str(kwargs) for kwargs in kwargs_rec)
    hover_scatter(elapsed, rel_err,names, plotfun=plt.loglog)
    plt.xlabel('time(s)')
    plt.ylabel('relative error')
    plt.axis('equal')
    plt.title("benchmark results for "+dict_res["title"])
