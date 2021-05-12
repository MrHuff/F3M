import falkon
from falkon.kernels import GaussianKernel
from typing import Optional, Union, Tuple, Dict
import torch
from falkon.options import BaseOptions, FalkonOptions
import functools
from FFM_classes import *

class bench_kernel_falkon:
    def __init__(self,sigma):
        self.ls = sigma**2

    def mmv(self,X1, X2, v, obj, out=None, params=None):
        input_device = X1.device
        d  = X1.shape[1]
        if not params.use_cpu:
            self.device = "cuda:0"
            if v is not None:
                v = v.to(self.device)
        else:
            print("FFM only available for GPU")
            raise NotImplementedError
        bench_obj = benchmark_matmul(X=X1,Y=X2,ls=self.ls)
        res = bench_obj.forward(bench_obj.X, bench_obj.Y, v).to(input_device)
        del bench_obj
        torch.cuda.empty_cache()
        return res
    def dmmv(self,X1, X2, v, w, obj, out=None, params=None):
        input_device = X1.device
        if not params.use_cpu:
            self.device = "cuda:0"
            if v is not None:
                v = v.to(self.device)
            if w is not None:
                w = w.to(self.device)
        else:
            print("FFM only available for GPU")
        bench_obj = benchmark_matmul(X=X1,Y=X2,ls=self.ls)
        if v is None:
            res=w
        else:
            if w is None:
                res = bench_obj.forward(bench_obj.X,bench_obj.Y,v)
            else:
                res = bench_obj.forward(bench_obj.X,bench_obj.Y,v) + w
        res_2 = bench_obj.forward(bench_obj.Y,bench_obj.X,res).to(input_device)
        del bench_obj
        torch.cuda.empty_cache()

        return res_2

class bench_GaussianKernel(GaussianKernel):
    def __init__(self, sigma: Union[float, torch.Tensor], opt: Optional[FalkonOptions] = None):
        super(bench_GaussianKernel, self).__init__(sigma, opt)
        self.kernel = bench_kernel_falkon(self.sigma)

    def _decide_mmv_impl(self, X1, X2, v, opt: FalkonOptions):
        return self.kernel.mmv

    def _decide_dmmv_impl(self, X1, X2, v, w, opt: FalkonOptions):
        return self.kernel.dmmv
