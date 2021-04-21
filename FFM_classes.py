import torch
from torch.utils.cpp_extension import load
load_obj = load(name='ffm_3d_float', sources=['pybinder_setup.cu'])

def calc_rel_error(true_res,approx_res):
    return torch.mean(torch.abs((true_res.squeeze()-approx_res.squeeze())/true_res.squeeze()))
class FFM:
    def __init__(self,
                 X,
                 Y=None,
                 ls=1.0,
                 min_points=1000,
                 nr_of_interpolation=64,
                 eff_var_limit=0.15,
                 var_compression=False,
                 smooth_interpolation=False,
                 device = "cuda:0"
                 ):
        self.X = X.to(device).float()
        if torch.is_tensor(Y):
            self.Y = Y.to(device).float()
        else:
            self.Y = Y
        self.d = self.X.shape[1]
        try:
            assert self.d>0 and self.d<6
        except AssertionError:
            print('Sorry bro, dimensionality of your data is too big; Can only do up to 5')
        self.ls = float(ls)
        self.min_points = float(min_points)
        self.nr_of_interpolation = int(nr_of_interpolation)
        self.eff_var_limit=float(eff_var_limit)
        self.var_compression = var_compression
        self.smooth_interpolation = smooth_interpolation
        self.device = device

    def update_ls(self,ls):
        try:
            assert ls>0
        except AssertionError:
            print('ls less than 0, not allowed')
        self.ls = float(ls)

    def __matmul__(self, b):
        b = b.to(self.device).float()
        if torch.is_tensor(self.Y):
            try:
                assert self.Y.shape[1]==self.X.shape[1]
                assert self.Y.shape[0]==b.shape[0]
            except AssertionError:
                print('hey check the shapes of your tensor X,Y and b they dont match up!')
            if self.d==1:
                return load_obj.FFM_XY_FLOAT_1(self.X,self.Y,b,self.device,self.ls,self.min_points,self.nr_of_interpolation)
            if self.d==2:
                return load_obj.FFM_XY_FLOAT_2(self.X,self.Y,b,self.device,self.ls,self.min_points,self.nr_of_interpolation)
            if self.d==3:
                return load_obj.FFM_XY_FLOAT_3(self.X,self.Y,b,self.device,self.ls,self.min_points,self.nr_of_interpolation)
            if self.d==4:
                return load_obj.FFM_XY_FLOAT_4(self.X,self.Y,b,self.device,self.ls,self.min_points,self.nr_of_interpolation)
            if self.d==5:
                return load_obj.FFM_XY_FLOAT_5(self.X,self.Y,b,self.device,self.ls,self.min_points,self.nr_of_interpolation)
            if self.d==6:
                return load_obj.FFM_XY_FLOAT_6(self.X,self.Y,b,self.device,self.ls,self.min_points,self.nr_of_interpolation)
            if self.d==7:
                return load_obj.FFM_XY_FLOAT_7(self.X,self.Y,b,self.device,self.ls,self.min_points,self.nr_of_interpolation)
            if self.d==8:
                return load_obj.FFM_XY_FLOAT_8(self.X,self.Y,b,self.device,self.ls,self.min_points,self.nr_of_interpolation)
            if self.d==9:
                return load_obj.FFM_XY_FLOAT_9(self.X,self.Y,b,self.device,self.ls,self.min_points,self.nr_of_interpolation)
            if self.d==10:
                return load_obj.FFM_XY_FLOAT_10(self.X,self.Y,b,self.device,self.ls,self.min_points,self.nr_of_interpolation)
        else:
            try:
                assert self.X.shape[0]==b.shape[0]
            except AssertionError:
                print('hey check the shapes of your tensor X and b they dont match up!')

            if not self.smooth_interpolation:
                if self.d==1:
                    return load_obj.FFM_X_FLOAT_1(self.X,b,self.device,self.ls,self.min_points,self.nr_of_interpolation,self.var_compression,self.eff_var_limit)
                if self.d==2:
                    return load_obj.FFM_X_FLOAT_2(self.X,b,self.device,self.ls,self.min_points,self.nr_of_interpolation,self.var_compression,self.eff_var_limit)
                if self.d==3:
                    return load_obj.FFM_X_FLOAT_3(self.X,b,self.device,self.ls,self.min_points,self.nr_of_interpolation,self.var_compression,self.eff_var_limit)
                if self.d==4:
                    return load_obj.FFM_X_FLOAT_4(self.X,b,self.device,self.ls,self.min_points,self.nr_of_interpolation,self.var_compression,self.eff_var_limit)
                if self.d==5:
                    return load_obj.FFM_X_FLOAT_5(self.X,b,self.device,self.ls,self.min_points,self.nr_of_interpolation,self.var_compression,self.eff_var_limit)
                if self.d==6:
                    return load_obj.FFM_X_FLOAT_6(self.X,b,self.device,self.ls,self.min_points,self.nr_of_interpolation,self.var_compression,self.eff_var_limit)
                if self.d==7:
                    return load_obj.FFM_X_FLOAT_7(self.X,b,self.device,self.ls,self.min_points,self.nr_of_interpolation,self.var_compression,self.eff_var_limit)
                if self.d==8:
                    return load_obj.FFM_X_FLOAT_8(self.X,b,self.device,self.ls,self.min_points,self.nr_of_interpolation,self.var_compression,self.eff_var_limit)
                if self.d==9:
                    return load_obj.FFM_X_FLOAT_9(self.X,b,self.device,self.ls,self.min_points,self.nr_of_interpolation,self.var_compression,self.eff_var_limit)
                if self.d==10:
                    return load_obj.FFM_X_FLOAT_10(self.X,b,self.device,self.ls,self.min_points,self.nr_of_interpolation,self.var_compression,self.eff_var_limit)
            else:
                if self.d==1:
                    return load_obj.SUPERSMOOTH_FFM_X_FLOAT_1(self.X,b,self.device,self.ls,self.min_points,self.nr_of_interpolation,self.eff_var_limit)
                if self.d==2:
                    return load_obj.SUPERSMOOTH_FFM_X_FLOAT_2(self.X,b,self.device,self.ls,self.min_points,self.nr_of_interpolation,self.eff_var_limit)
                if self.d==3:
                    return load_obj.SUPERSMOOTH_FFM_X_FLOAT_3(self.X,b,self.device,self.ls,self.min_points,self.nr_of_interpolation,self.eff_var_limit)
                if self.d==4:
                    return load_obj.SUPERSMOOTH_FFM_X_FLOAT_4(self.X,b,self.device,self.ls,self.min_points,self.nr_of_interpolation,self.eff_var_limit)
                if self.d==5:
                    return load_obj.SUPERSMOOTH_FFM_X_FLOAT_5(self.X,b,self.device,self.ls,self.min_points,self.nr_of_interpolation,self.eff_var_limit)
                if self.d==6:
                    return load_obj.SUPERSMOOTH_FFM_X_FLOAT_6(self.X,b,self.device,self.ls,self.min_points,self.nr_of_interpolation,self.eff_var_limit)
                if self.d==7:
                    return load_obj.SUPERSMOOTH_FFM_X_FLOAT_7(self.X,b,self.device,self.ls,self.min_points,self.nr_of_interpolation,self.eff_var_limit)
                if self.d==8:
                    return load_obj.SUPERSMOOTH_FFM_X_FLOAT_8(self.X,b,self.device,self.ls,self.min_points,self.nr_of_interpolation,self.eff_var_limit)
                if self.d==9:
                    return load_obj.SUPERSMOOTH_FFM_X_FLOAT_9(self.X,b,self.device,self.ls,self.min_points,self.nr_of_interpolation,self.eff_var_limit)
                if self.d==10:
                    return load_obj.SUPERSMOOTH_FFM_X_FLOAT_10(self.X,b,self.device,self.ls,self.min_points,self.nr_of_interpolation,self.eff_var_limit)

class benchmark_matmul():
    def __init__(self,
                 X,
                 Y=None,
                 ls=1.0,
                 device = "cuda:0"):
        self.X = X.to(device).float()
        if torch.is_tensor(Y):
            self.Y = Y.to(device).float()
        else:
            self.Y = Y
        self.d = self.X.shape[1]
        try:
            assert self.d>0 and self.d<6
        except AssertionError:
            print('Sorry bro, dimensionality of your data is too big; Can only do up to 5')
        self.ls = float(ls)
        self.device = device

    def update_ls(self,ls):
        try:
            assert ls>0
        except AssertionError:
            print('ls less than 0, not allowed')
        self.ls = float(ls)


    def __matmul__(self, b):
        b = b.to(self.device).float()
        if torch.is_tensor(self.Y):
            try:
                assert self.Y.shape[1]==self.X.shape[1]
                assert self.Y.shape[0]==b.shape[0]
            except AssertionError:
                print('hey check the shapes of your tensor X,Y and b they dont match up!')

            Y=self.Y
        else:
            Y=self.X
            try:
                assert self.X.shape[0]==b.shape[0]
            except AssertionError:
                print('hey check the shapes of your tensor X and b they dont match up!')
        if self.d==1:
            return load_obj.rbf_float_1(self.X,Y,b,self.ls,True)
        if self.d==2:
            return load_obj.rbf_float_2(self.X,Y,b,self.ls,True)
        if self.d==3:
            return load_obj.rbf_float_3(self.X,Y,b,self.ls,True)
        if self.d==4:
            return load_obj.rbf_float_4(self.X,Y,b,self.ls,True)
        if self.d==5:
            return load_obj.rbf_float_5(self.X,Y,b,self.ls,True)
        if self.d==6:
            return load_obj.rbf_float_6(self.X,Y,b,self.ls,True)
        if self.d==7:
            return load_obj.rbf_float_7(self.X,Y,b,self.ls,True)
        if self.d==8:
            return load_obj.rbf_float_8(self.X,Y,b,self.ls,True)
        if self.d==9:
            return load_obj.rbf_float_9(self.X,Y,b,self.ls,True)
        if self.d==10:
            return load_obj.rbf_float_10(self.X,Y,b,self.ls,True)






