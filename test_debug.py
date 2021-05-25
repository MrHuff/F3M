
# def divide(old_interactions, depth, p):
#     n = old_interactions.shape[0]
#     arr = torch.arange(p)
#     if depth==0:
#         left = arr.repeat_interleave(p).repeat(n)
#         right = arr.repeat(p*n)
#         add = p * old_interactions.repeat_interleave(p * p, 0)
#         new_interations = torch.stack([left,right],dim=1
#                                       ) + add
#     else:
#         left_old,right_old = old_interactions.unbind(1)
#         output, counts = torch.unique_consecutive(left_old,return_counts=True)
#         right = arr.repeat(p*n)+right_old.repeat_interleave(p)
#         left = arr.repeat_interleave(p).repeat(output.shape[0]).repeat_interleave(counts.repeat_interleave(p*p))+p*left_old.repeat_interleave(p*p,0)
#         new_interations = torch.stack([left,right],dim=1
#                                       )
#     return new_interations
#
# p=4
# interactions = torch.tensor([[0,0]])
# interactions = divide(interactions,1,p)
# print(interactions)
# interactions = interactions[torch.rand(interactions.shape[0])>0.4]
# print(interactions)
# interactions = divide(interactions,2,p)
# print(interactions)
# interactions = interactions[torch.rand(interactions.shape[0])>0.4]
# print(interactions)



import torch
#
from FFM_classes import *
import time
from run_obj import *


# import pykeops
# pykeops.clean_pykeops()

if __name__ == '__main__':
    N = 1000000000
    d=3
    m = 100000
    X = torch.randn(N, d).float()
    # b = torch.randn(N, 1).float()
    b = torch.randn(m, 1).float()
    x_ref = X[:m].float()
    ls = 1
    min_points = 10000
    small_field = 64
    # obj_test = FFM(X=X,ls=ls,min_points=min_points,eff_var_limit=0.1,var_compression=True,small_field_points=64)
    # start= time.time()
    # obj_test@b
    # end = time.time()
    # sq_time =end-start
    # print(sq_time)
    # feels very much like a cache/loading issue??!?!?!?!?!
    # timing doesn't work either...
    # obj_test = FFM(X=x_ref,Y=X,ls=ls,min_points=min_points,eff_var_limit=0.1,var_compression=True,small_field_points=small_field)
    # start= time.time()
    # f_1 = obj_test@b
    # end = time.time()
    # asymetric_time =end-start
    # print(asymetric_time)

    obj_test_2 = FFM(X=X,Y=x_ref,ls=ls,min_points=min_points,eff_var_limit=0.1,var_compression=True,small_field_points=small_field)
    start= time.time()
    f_2 = obj_test_2@b
    end = time.time()
    transpose =end-start
    print(transpose)

    # bmark_1 = keops_matmul(X=x_ref,Y=X,ls=ls,type=torch.float32)
    # bmark_2 = benchmark_matmul(X=x_ref,Y=X,ls=ls)
    # #
    # #
    # start= time.time()
    # res = bmark_2@b
    # end = time.time()
    # keops_full =end-start
    # print(keops_full)
    # print(torch.norm(res-f_1)/torch.norm(res))

    #
    # start= time.time()
    # bmark_1@b
    # end = time.time()
    # keops_assym =end-start
    # # print('sq_time: ',sq_time)
    # print('transpose_time: ',transpose)
    # # print('keops_full: ',keops_full)
    # print('keops_assym: ',keops_assym)


# # torch.multiprocessing.set_start_method('spawn')# good solution !!!!
#     # N = 100000000
#     # d = 3
#     # ls = 3.0
#     # penalty = 1e-5
#     # M = 10000
#     # X = torch.rand(M, d)
#     # b = torch.randn(N, 1)
#     # Y = torch.randn(N,d)
#     # min_points = 64
#     # obj_test = par_FFM(X=X,Y=Y,ls=ls,min_points=min_points,eff_var_limit=0.1,var_compression=False,small_field_points=64,par_factor=2)
#     # # obj_test = FFM(X=X,Y=Y,ls=ls,min_points=min_points,eff_var_limit=0.1,var_compression=False,small_field_points=64)
#     # start = time.time()
#     # obj_test@b
#     # end = time.time()
#     # print(end-start)
#
#
#
#
#
#
