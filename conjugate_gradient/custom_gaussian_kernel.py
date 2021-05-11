import falkon
from falkon.kernels import GaussianKernel
from typing import Optional, Union, Tuple, Dict
import torch
from falkon.options import BaseOptions, FalkonOptions
import functools
from FFM_classes import *

class FFM_kernel_falkon:
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
        FFM_obj = FFM(X=X1,Y=X2, ls=self.ls, min_points=2500, nr_of_interpolation=4**d,
                      eff_var_limit=0.1, var_compression=True,  device=self.device,
                      small_field_points=4**d)
        res = FFM_obj.forward(FFM_obj.X, FFM_obj.Y, v).to(input_device)
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
        d  = X1.shape[1]
        FFM_obj = FFM(X=X1,Y=X2, ls=self.ls, min_points=2500, nr_of_interpolation=4**d,
                      eff_var_limit=0.1, var_compression=True,device=self.device,
                      small_field_points=4**d)
        if v is None:
            res=w
        else:
            if w is None:
                res = FFM_obj.forward(FFM_obj.X,FFM_obj.Y,v)
            else:
                res = FFM_obj.forward(FFM_obj.X,FFM_obj.Y,v) + w
        res_2 = FFM_obj.forward(FFM_obj.Y,FFM_obj.X,res).to(input_device)
        del FFM_obj
        torch.cuda.empty_cache()

        return res_2

class custom_GaussianKernel(GaussianKernel):
    def __init__(self, sigma: Union[float, torch.Tensor], opt: Optional[FalkonOptions] = None):
        super(custom_GaussianKernel, self).__init__(sigma,opt)
        self.kernel = FFM_kernel_falkon(self.sigma)

    def _decide_mmv_impl(self, X1, X2, v, opt: FalkonOptions):
        return self.kernel.mmv

    def _decide_dmmv_impl(self, X1, X2, v, w, opt: FalkonOptions):
        return self.kernel.dmmv
