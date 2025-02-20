import torch
import time
from run_obj import *


def chunk_forward(FFM_obj,X,Y,b,chunks=10):
    b_chunked = torch.chunk(b,dim=0,chunks=chunks)
    Y_chunked = torch.chunk(Y,dim=0,chunks=chunks)
    X = X.to(FFM_obj.device)
    res = 0.0
    for b,y in zip(b_chunked,Y_chunked):
        b,y = b.to(FFM_obj.device),y.to(FFM_obj.device)
        res+=FFM_obj.forward(X,y,b)
    return res

if __name__ == '__main__':
    M=100000
    ls=1.4
    N = 1000000000
    nodes = 64
    problem_set = torch.load('krr_taxi.pt')
    X = problem_set['X'][:N, :].contiguous()
    for seed in [1]:
        Y = torch.randn(N,1)
        X_t = X[:M,:]
        Y_t = torch.randn(M,1)

        obj_test = FFM(X=X_t,Y=X,ls=ls,min_points=1000,eff_var_limit=2.0,var_compression=True,small_field_points=nodes,nr_of_interpolation=nodes)
        # obj_test_t = FFM(X=X,Y=X_t,ls=ls,min_points=250,eff_var_limit=0.3,var_compression=True,small_field_points=nodes,nr_of_interpolation=nodes)
        bmark = benchmark_matmul(X=X_t[:3,:],Y=X[:3,:],ls=ls)
        # bmark_t = benchmark_matmul(X=X,Y=X_t,ls=ls)

        start= time.time()
        res_0 = obj_test.chunk_forward_example(Y,chunks=20)
        end = time.time()
        time_0 = end-start


        # start= time.time()
        # res_1 = obj_test@Y
        # end = time.time()
        # time_1 = end-start
        #
        # start= time.time()
        # res_2 = obj_test_t@Y_t
        # end = time.time()
        # time_2 = end-start
        #
        start= time.time()
        res_3 = bmark.forward(obj_test.X,obj_test.Y,obj_test.b)
        end = time.time()
        time_3 = end-start
        #
        # start= time.time()
        # res_4 = bmark_t@Y_t
        # end = time.time()
        # time_4 = end-start
        print(time_0)
        # print(time_1)
        # print(time_2)
        # print(time_3)
        # print(time_4)
        #
        print(torch.norm(res_0[:M]-res_3)/torch.norm(res_3))
        # print(torch.norm(res_1[:M]-res_3)/torch.norm(res_3))
        # print(torch.norm(res_2-res_4)/torch.norm(res_4))
        torch.cuda.empty_cache()





