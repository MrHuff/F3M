import torch
import os
from torch.utils.cpp_extension import load
from pykeops.torch import Genred
import torch.multiprocessing as mp
import ffm_3d_float as load_obj

class FFM:
    def __init__(self,
                 X,
                 Y=None,
                 ls=1.0,
                 min_points=1000,
                 nr_of_interpolation=64,
                 eff_var_limit=0.15,
                 var_compression=False,
                 small_field_points = 1000,
                 device = "cuda:0"
                 ):

        self.X = X.float().to(device)
        if torch.is_tensor(Y):
            self.Y = Y.float().to(device)
        else:
            self.Y = self.X
            print('X==Y assuming kernel covariance matmul')
            assert self.X.data_ptr() == self.Y.data_ptr()
        self.d = self.X.shape[1]
        try:
            assert self.d<10
        except AssertionError:
            print('Sorry bro, dimensionality of your data is too big; Can only do up to 10')
        self.ls = float(ls)
        self.min_points = float(min_points)
        self.nr_of_interpolation = int(nr_of_interpolation)
        self.eff_var_limit=float(eff_var_limit)
        self.var_compression = var_compression
        self.device = device
        self.small_field_points = small_field_points

    def update_ls(self,ls):
        try:
            assert ls>0
        except AssertionError:
            print('ls less than 0, not allowed')
        self.ls = float(ls)

    def __matmul__(self, b):
        self.b = b.float().to(self.device)
        return self.forward(self.X,self.Y,self.b)
    def grad(self,b):
        self.b = b.float().to(self.device)
        return self.forward_grad(self.X,self.Y,self.b)
    def chunk_forward_transpose(self,b,chunks=10):
        self.b = b.float().to(self.device)
        b_chunked = torch.chunk(self.b, dim=0, chunks=chunks)
        Y_chunked = torch.chunk(self.X, dim=0, chunks=chunks)
        res = 0.0
        for b, y in zip(b_chunked, Y_chunked):
            res += self.forward(self.Y, y, b)
        return res

    def chunk_forward_example(self,b,chunks=10):
        self.b = b.float().to(self.device)
        b_chunked = torch.chunk(self.b, dim=0, chunks=chunks)
        Y_chunked = torch.chunk(self.Y, dim=0, chunks=chunks)
        res = 0.0
        for b, y in zip(b_chunked, Y_chunked):
            res += self.forward(self.X, y, b)
        return res
    def forward_grad(self,X,Y,b):
        try:
            assert Y.shape[1]==X.shape[1]
            assert Y.shape[0]==b.shape[0]
        except AssertionError:
            print('hey check the shapes of your tensor X,Y and b they dont match up!')
            raise AssertionError
        if self.d==1:
            return load_obj.rbf_float_1_grad(X,Y,b,self.ls,True)
        if self.d==2:
            return load_obj.rbf_float_2_grad(X,Y,b,self.ls,True)
        if self.d==3:
            return load_obj.rbf_float_3_grad(X,Y,b,self.ls,True)
        if self.d==4:
            return load_obj.rbf_float_4_grad(X,Y,b,self.ls,True)
        if self.d==5:
            return load_obj.rbf_float_5_grad(X,Y,b,self.ls,True)
        if self.d==6:
            return load_obj.rbf_float_6_grad(X,Y,b,self.ls,True)
        if self.d==7:
            return load_obj.rbf_float_7_grad(X,Y,b,self.ls,True)
        if self.d==8:
            return load_obj.rbf_float_8_grad(X,Y,b,self.ls,True)
        if self.d==9:
            return load_obj.rbf_float_9_grad(X,Y,b,self.ls,True)
        if self.d==10:
            return load_obj.rbf_float_10_grad(X,Y,b,self.ls,True)
    def forward(self,X,Y,b):
        assert X.device==Y.device==b.device==torch.device(self.device)
        self.device = str(self.device)
        try:
            assert Y.shape[1]==X.shape[1]
            assert Y.shape[0]==b.shape[0]
        except AssertionError:
            print('hey check the shapes of your tensor X,Y and b they dont match up!')
            raise AssertionError
        if self.d==1:
            return load_obj.FFM_XY_FLOAT_1(X,Y,b,self.device,self.ls,self.min_points,self.nr_of_interpolation,self.var_compression,self.eff_var_limit,self.small_field_points)
        if self.d==2:
            return load_obj.FFM_XY_FLOAT_2(X,Y,b,self.device,self.ls,self.min_points,self.nr_of_interpolation,self.var_compression,self.eff_var_limit,self.small_field_points)
        if self.d==3:
            return load_obj.FFM_XY_FLOAT_3(X,Y,b,self.device,self.ls,self.min_points,self.nr_of_interpolation,self.var_compression,self.eff_var_limit,self.small_field_points)
        if self.d==4:
            return load_obj.FFM_XY_FLOAT_4(X,Y,b,self.device,self.ls,self.min_points,self.nr_of_interpolation,self.var_compression,self.eff_var_limit,self.small_field_points)
        if self.d==5:
            return load_obj.FFM_XY_FLOAT_5(X,Y,b,self.device,self.ls,self.min_points,self.nr_of_interpolation,self.var_compression,self.eff_var_limit,self.small_field_points)
        if self.d==6:
            return load_obj.FFM_XY_FLOAT_6(X,Y,b,self.device,self.ls,self.min_points,self.nr_of_interpolation,self.var_compression,self.eff_var_limit,self.small_field_points)
        if self.d==7:
            return load_obj.FFM_XY_FLOAT_7(X,Y,b,self.device,self.ls,self.min_points,self.nr_of_interpolation,self.var_compression,self.eff_var_limit,self.small_field_points)
        if self.d==8:
            return load_obj.FFM_XY_FLOAT_8(X,Y,b,self.device,self.ls,self.min_points,self.nr_of_interpolation,self.var_compression,self.eff_var_limit,self.small_field_points)
        if self.d==9:
            return load_obj.FFM_XY_FLOAT_9(X,Y,b,self.device,self.ls,self.min_points,self.nr_of_interpolation,self.var_compression,self.eff_var_limit,self.small_field_points)
        if self.d==10:
            return load_obj.FFM_XY_FLOAT_10(X,Y,b,self.device,self.ls,self.min_points,self.nr_of_interpolation,self.var_compression,self.eff_var_limit,self.small_field_points)

class par_FFM_block():
    def __init__(self,
                 X,
                 b,
                 Y=None,
                 ls=1.0,
                 min_points=1000,
                 nr_of_interpolation=64,
                 eff_var_limit=0.15,
                 var_compression=False,
                 small_field_points=1000,
                 devices=[0]
                 ):
        self.X = X.float()
        self.b = b.float()
        if torch.is_tensor(Y):
            self.Y = Y.float()
        else:
            self.Y = self.X
            print('X==Y assuming kernel covariance matmul')
            assert self.X.data_ptr() == self.Y.data_ptr()
        self.d = self.X.shape[1]
        try:
            assert self.d < 6
        except AssertionError:
            print('Sorry bro, dimensionality of your data is too big; Can only do up to 5')
        self.ls = float(ls)
        self.min_points = float(min_points)
        self.nr_of_interpolation = int(nr_of_interpolation)
        self.eff_var_limit = float(eff_var_limit)
        self.var_compression = var_compression
        self.devices = devices
        self.small_field_points = small_field_points
        self.par_fac = len(devices)
        print(self.par_fac)
        y_chunks = torch.chunk(self.Y, self.par_fac, dim=0)
        x_chunks = torch.chunk(self.X, self.par_fac, dim=0)
        b_chunks = torch.chunk(self.b, self.par_fac, dim=0)
        self.jobs_list = []
        for y_sub,b_sub in zip(y_chunks,b_chunks):
            for x_sub,device in zip(x_chunks,self.devices):
                self.jobs_list.append((x_sub.clone(),y_sub.clone(),b_sub.clone(),device))

    def __call__(self):
        with mp.Pool(processes = self.par_fac) as p:   # Paralleizing over 2 GPUs
            results = p.starmap(self.forward,self.jobs_list)
        return results

    def update_ls(self,ls):
        try:
            assert ls>0
        except AssertionError:
            print('ls less than 0, not allowed')
        self.ls = float(ls)


    def forward(self, X, Y, b,device):
        print(device)
        os.environ["CUDA_VISIBLE_DEVICES"] = str(device)
        X = X.to("cuda:0")
        Y = Y.to("cuda:0")
        b = b.to("cuda:0")
        assert X.device == Y.device == b.device == torch.device(0)
        try:
            assert Y.shape[1] == X.shape[1]
            assert Y.shape[0] == b.shape[0]
        except AssertionError:
            print('hey check the shapes of your tensor X,Y and b they dont match up!')
            raise AssertionError
        torch.cuda.synchronize()

        if self.d == 1:
            output = load_obj.FFM_XY_FLOAT_1(X, Y, b, "cuda:0", self.ls, self.min_points, self.nr_of_interpolation,
                                           self.var_compression, self.eff_var_limit, self.small_field_points)
        if self.d == 2:
            output = load_obj.FFM_XY_FLOAT_2(X, Y, b, "cuda:0", self.ls, self.min_points, self.nr_of_interpolation,
                                           self.var_compression, self.eff_var_limit, self.small_field_points)
        if self.d == 3:
            output = load_obj.FFM_XY_FLOAT_3(X, Y, b, "cuda:0", self.ls, self.min_points, self.nr_of_interpolation,
                                           self.var_compression, self.eff_var_limit, self.small_field_points)
        torch.cuda.synchronize()
        del X,Y,b
        torch.cuda.empty_cache()
        return output

class keops_matmul():
    def __init__(self,
                 X,
                 Y=None,
                 ls=1.0,
                 device = "cuda:0",
                 type=torch.float32):
        self.type=type
        self.device =device
        self.X = X.type(self.type).to(device)
        self.D = X.shape[1]
        formula = 'Exp(IntInv(-2) * SqDist(x / g, y / g)) * b'
        aliases = [
            "x = Vi(" + str(self.D) + ")",  # First arg:  i-variable of size D
            "y = Vj(" + str(self.D) + ")",  # Second arg: j-variable of size D
            "b = Vj(" + str(1) + ")",  # Third arg:  j-variable of size Dv
            "g = Pm(1)",
        ]

        self.red_func = Genred(formula,  # F(g,x,y,b) = exp( -g*|x-y|^2 ) * b
                              aliases,  # Fourth arg is indexed by "j", of dim 2
                               reduction_op='Sum',
                               axis=1,
                               dtype="float32")
        if torch.is_tensor(Y):
            self.Y = Y.type(self.type).to(device)
        else:
            self.Y = self.X
        self.d = self.X.shape[1]
        try:
            assert self.d > 0 and self.d < 10
        except AssertionError:
            print('Sorry bro, dimensionality of your data is too big; Can only do up to 10')
        self.ls = torch.tensor([ls**0.5]).type(self.type).to(self.device)
        self.device = device

    def __matmul__(self, b):
        self.b = b.type(self.type).to(self.device)
        return self.forward(self.X, self.Y, self.b)

    def forward(self, X, Y, b):
        assert X.device == Y.device == b.device == torch.device(self.device)
        self.device = str(self.device)
        try:
            assert Y.shape[1] == X.shape[1]
            assert Y.shape[0] == b.shape[0]
        except AssertionError:
            print('hey check the shapes of your tensor X,Y and b they dont match up!')
            raise AssertionError
        return self.red_func(X,Y,b,self.ls).float()

class benchmark_matmul():
    def __init__(self,
                 X,
                 Y=None,
                 ls=1.0,
                 device = "cuda:0"):
        self.X = X.float().to(device)
        if torch.is_tensor(Y):
            self.Y = Y.float().to(device)
        else:
            self.Y = self.X
        self.d = self.X.shape[1]
        try:
            assert self.d>0 and self.d<10
        except AssertionError:
            print('Sorry bro, dimensionality of your data is too big; Can only do up to 10')
            raise AssertionError
        self.ls = float(ls)
        self.device = device

    def update_ls(self,ls):
        try:
            assert ls>0
        except AssertionError:
            print('ls less than 0, not allowed')
            raise AssertionError

        self.ls = float(ls)

    def __matmul__(self, b):
        self.b = b.float().to(self.device)
        return self.forward(self.X,self.Y,self.b)

    def forward(self,X,Y,b):
        try:
            assert Y.shape[1]==X.shape[1]
            assert Y.shape[0]==b.shape[0]
        except AssertionError:
            print('hey check the shapes of your tensor X,Y and b they dont match up!')
            raise AssertionError
        if self.d==1:
            return load_obj.rbf_float_1(X,Y,b,self.ls,True)
        if self.d==2:
            return load_obj.rbf_float_2(X,Y,b,self.ls,True)
        if self.d==3:
            return load_obj.rbf_float_3(X,Y,b,self.ls,True)
        if self.d==4:
            return load_obj.rbf_float_4(X,Y,b,self.ls,True)
        if self.d==5:
            return load_obj.rbf_float_5(X,Y,b,self.ls,True)
        if self.d==6:
            return load_obj.rbf_float_6(X,Y,b,self.ls,True)
        if self.d==7:
            return load_obj.rbf_float_7(X,Y,b,self.ls,True)
        if self.d==8:
            return load_obj.rbf_float_8(X,Y,b,self.ls,True)
        if self.d==9:
            return load_obj.rbf_float_9(X,Y,b,self.ls,True)
        if self.d==10:
            return load_obj.rbf_float_10(X,Y,b,self.ls,True)

class benchmark_matmul_double():
    def __init__(self,
                 X,
                 Y=None,
                 ls=1.0,
                 device = "cuda:0"):
        self.X = X.double().to(device)
        if torch.is_tensor(Y):
            self.Y = Y.double().to(device)
        else:
            self.Y = self.X
        self.d = self.X.shape[1]
        try:
            assert self.d>0 and self.d<10
        except AssertionError:
            print('Sorry bro, dimensionality of your data is too big; Can only do up to 10')
            raise AssertionError

        self.ls = float(ls)
        self.device = device

    def update_ls(self,ls):
        try:
            assert ls>0
        except AssertionError:
            print('ls less than 0, not allowed')
            raise AssertionError

        self.ls = float(ls)

    def __matmul__(self, b):
        self.b = b.double().to(self.device)
        try:
            assert self.Y.shape[1]==self.X.shape[1]
            assert self.Y.shape[0]==b.shape[0]
        except AssertionError:
            print('hey check the shapes of your tensor X,Y and b they dont match up!')
            raise AssertionError

        if self.d==1:
            return load_obj.rbf_double_1(self.X,self.Y,self.b,self.ls,True)
        if self.d==2:
            return load_obj.rbf_double_2(self.X,self.Y,self.b,self.ls,True)
        if self.d==3:
            return load_obj.rbf_double_3(self.X,self.Y,self.b,self.ls,True)
        if self.d==4:
            return load_obj.rbf_double_4(self.X,self.Y,self.b,self.ls,True)
        if self.d==5:
            return load_obj.rbf_double_5(self.X,self.Y,self.b,self.ls,True)
        if self.d==6:
            return load_obj.rbf_double_6(self.X,self.Y,self.b,self.ls,True)
        if self.d==7:
            return load_obj.rbf_double_7(self.X,self.Y,self.b,self.ls,True)
        if self.d==8:
            return load_obj.rbf_double_8(self.X,self.Y,self.b,self.ls,True)
        if self.d==9:
            return load_obj.rbf_double_9(self.X,self.Y,self.b,self.ls,True)
        if self.d==10:
            return load_obj.rbf_double_10(self.X,self.Y,self.b,self.ls,True)

class block_FFM_matmul(FFM):
    def __init__(self,
                 X,
                 Y=None,
                 ls=1.0,
                 min_points=1000,
                 nr_of_interpolation=64,
                 eff_var_limit=0.15,
                 var_compression=False,
                 small_field_points = 1000,
                 device = "cuda:0",
                 blocks_X=1,
                 blocks_Y=1
                 ):
        super(block_FFM_matmul, self).__init__(X=X,
                                               Y=Y,
                                               ls=ls,
                                               min_points=min_points,
                                               nr_of_interpolation=nr_of_interpolation,
                                               eff_var_limit=eff_var_limit,
                                               var_compression=var_compression,
                                               small_field_points=small_field_points,
                                               device=device
                                               )
        self.blocks_X = blocks_X
        self.blocks_Y = blocks_Y

    def __matmul__(self, b):
        output_list = []
        if torch.is_tensor(self.Y): ##XY MODE
            y_chunks = torch.chunk(self.Y,self.blocks_Y,dim=0)
            x_chunks = torch.chunk(self.X,self.blocks_X,dim=0)
            b_chunks = torch.chunk(b,self.blocks_Y,dim=0)
            for i, x in enumerate(x_chunks):
                output_append = torch.zeros(x.shape[0],b.shape[1])
                for j, y in  enumerate(y_chunks):
                    output_append+=self.forward(x,y,b_chunks[j]).cpu()
                output_list.append(output_append)
            return torch.cat(output_list,dim=0)
        else: #XX MODE
            x_chunks = torch.chunk(self.X, self.blocks_X, dim=0)
            b_chunks = torch.chunk(b, self.blocks_X, dim=0)
            for i,x in enumerate(x_chunks):
                output_append = torch.zeros(x.shape[0], b.shape[1])
                for j, y  in enumerate(x_chunks):
                    if i==j:
                        output_append += self.forward(x, None, b_chunks[j]).cpu()
                    else:
                        output_append += self.forward(x, y, b_chunks[j]).cpu()
                output_list.append(output_append)
            return torch.cat(output_list,dim=0)












