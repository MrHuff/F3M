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
        self.device  = X1.device
        FFM_obj = FFM(X=X1,Y=X2, ls=self.ls, min_points=2500, nr_of_interpolation=64,
                      eff_var_limit=0.1, var_compression=True, smooth_interpolation=False, device=self.device,
                      small_field_points=64)
        return FFM_obj@v

    def dmmv(self,X1, X2, v, w, obj, out=None, params=None):
        self.device  = X1.device.type + ':'+str(X1.device.index)

        FFM_obj = FFM(X=X1,Y=X2, ls=self.ls, min_points=100, nr_of_interpolation=64,
                      eff_var_limit=0.1, var_compression=True, smooth_interpolation=False, device=self.device,
                      small_field_points=64)
        if v is None:
            res=w
        else:
            if w is None:
                res = FFM_obj.forward(X1,X2,v)
            else:
                res = FFM_obj.forward(X1,X2,v) + w


        res_2 = FFM_obj.forward(X2,X1,res)
        return res_2

class custom_GaussianKernel(GaussianKernel):
    def __init__(self, sigma: Union[float, torch.Tensor], opt: Optional[FalkonOptions] = None):
        super(custom_GaussianKernel, self).__init__(sigma,opt)
        self.kernel = FFM_kernel_falkon(self.sigma)

    def _decide_mmv_impl(self, X1, X2, v, opt: FalkonOptions):
        return self.kernel.mmv

    def _decide_dmmv_impl(self, X1, X2, v, w, opt: FalkonOptions):
        return self.kernel.dmmv
