import falkon
from falkon.kernels import GaussianKernel
from typing import Optional, Union, Tuple, Dict
import torch
from falkon.options import BaseOptions, FalkonOptions
import functools
from FFM_classes import *


class bench_GaussianKernel(GaussianKernel):
    def __init__(self, sigma: Union[float, torch.Tensor], opt: Optional[FalkonOptions] = None,
                 min_points: [float] = None):
        super(bench_GaussianKernel, self).__init__(sigma, opt)
        self.ls = sigma ** 2
        self.min_points = min_points
        self.ffm_initialized = False
        self.device = "cuda:0"

    def _decide_mmv_impl(self, X1, X2, v, opt: FalkonOptions):
        if not self.ffm_initialized:
            d = X1.shape[1]
            if self.min_points is None:
                self.min_points = 4 ** d
            self.bench_obj = benchmark_matmul(X=X1, Y=X2, ls=self.ls, device=self.device)
            self.ffm_initialized = True

        return self.mmv_

    def _decide_dmmv_impl(self, X1, X2, v, w, opt: FalkonOptions):
        if not self.ffm_initialized:
            d = X1.shape[1]
            if self.min_points is None:
                self.min_points = 4 ** d
            self.bench_obj =benchmark_matmul(X=X1, Y=X2, ls=self.ls, device=self.device)
            self.ffm_initialized = True

        return self.dmmv_

    def mmv_(self, X1, X2, v, obj, out=None, params=None):
        input_device = X1.device
        if not params.use_cpu:
            self.device = "cuda:0"
            if v is not None:
                v = v.to(self.device)
        else:
            print("FFM only available for GPU")
            raise NotImplementedError

        res = self.bench_obj.forward(self.bench_obj.X, self.bench_obj.Y, v).to(input_device)
        torch.cuda.empty_cache()
        return res

    def dmmv_(self, X1, X2, v, w, obj, out=None, params=None):
        input_device = X1.device
        if not params.use_cpu:
            self.device = "cuda:0"
            if v is not None:
                v = v.to(self.device)
            if w is not None:
                w = w.to(self.device)
        else:
            print("FFM only available for GPU")

        if v is None:
            res = w
        else:
            if w is None:
                res = self.bench_obj.forward(self.bench_obj.X, self.bench_obj.Y, v)
            else:
                res = self.bench_obj.forward(self.bench_obj.X, self.bench_obj.Y, v) + w
        res_2 = self.bench_obj.forward(self.bench_obj.Y, self.bench_obj.X, res).to(input_device)
        torch.cuda.empty_cache()

        return res_2