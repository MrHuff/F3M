import falkon
from falkon.kernels import GaussianKernel
from typing import Optional, Union, Tuple, Dict
import torch
from falkon.options import BaseOptions, FalkonOptions
import functools
from FFM_classes import *
import time
class custom_GaussianKernel(GaussianKernel):
    def __init__(self, sigma: Union[float, torch.Tensor], opt: Optional[FalkonOptions] = None,min_points: [float]=None,var_compression: [bool]=True,interpolation_nr: [int]=64,eff_var_limit: [float]=0.3):
        super(custom_GaussianKernel, self).__init__(sigma,opt)
        self.ls = sigma
        self.min_points = min_points
        self.ffm_initialized = False
        self.device = "cuda:0"
        self.interpolation_nr = interpolation_nr
        self.var_compression = var_compression
        self.eff_var_limit = eff_var_limit
    def _decide_mmv_impl(self, X1, X2, v, opt: FalkonOptions):
        if not self.ffm_initialized:
            d = X1.shape[1]
            if self.min_points is None:
                self.min_points = 4 ** d
            self.FFM_obj = FFM(X=X1, Y=X2, ls=self.ls, min_points=self.min_points, nr_of_interpolation=self.interpolation_nr,
                          eff_var_limit=self.eff_var_limit, var_compression=self.var_compression, device=self.device,
                          small_field_points=self.interpolation_nr)
            self.ffm_initialized = True

        return self.mmv_

    def _decide_dmmv_impl(self, X1, X2, v, w, opt: FalkonOptions):
        if not self.ffm_initialized:
            d = X1.shape[1]
            if self.min_points is None:
                self.min_points = 4 ** d
            self.FFM_obj = FFM(X=X1, Y=X2, ls=self.ls, min_points=self.min_points, nr_of_interpolation=self.interpolation_nr,
                          eff_var_limit=self.eff_var_limit, var_compression=self.var_compression, device=self.device,
                          small_field_points=self.interpolation_nr)
            self.ffm_initialized = True

        return self.dmmv_
    
    def mmv_(self,X1, X2, v, obj, out=None, params=None):
        input_device = X1.device
        if not params.use_cpu:
            self.device = "cuda:0"
            if v is not None:
                v = v.to(self.device)
        else:
            print("FFM only available for GPU")
            raise NotImplementedError
        res = self.FFM_obj.forward(self.FFM_obj.X, self.FFM_obj.Y, v)
        return res.to(input_device)
    
    def dmmv_(self,X1, X2, v, w, obj, out=None, params=None):
        input_device = X1.device
        if not params.use_cpu:
            self.device = "cuda:0"
            if v is not None:
                v = v.to(self.device)
            if w is not None:
                w = w.to(self.device)
        else:
            print("FFM only available for GPU")
        s_all = time.time()
        if v is None:
            res=w
        else:
            if w is None:
                start=time.time()
                res = self.FFM_obj.forward(self.FFM_obj.X,self.FFM_obj.Y,v)
                end = time.time()
                print("FFM actual time 1 (1) : ", end-start)
            else:
                start=time.time()
                res = self.FFM_obj.forward(self.FFM_obj.X,self.FFM_obj.Y,v) + w
                end = time.time()
                print("FFM actual time 1 (2) : ", end-start)
        start=time.time()
        #weirdest thing ever... why is it so slow on consequtive loads???
        res_2 = self.FFM_obj.forward(self.FFM_obj.Y,self.FFM_obj.X,res)
        end = time.time()
        print("FFM actual time 2 : ", end-start)
        print("total FFM time: ", end-s_all)
        return res_2.to(input_device)