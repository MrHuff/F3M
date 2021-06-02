import torch

from FFM_classes import *
import numpy as np
torch.manual_seed(0)
d=3
N=1000000000
# for seed in [0.1,1,10]:
#     ls = 1/(2**0.5*seed**0.5)
#     perm = torch.arange(0,1000)
#     alpha = torch.randn(1000,1)
#     X_1= torch.rand(N,d)
#     x_ref_1 = X_1[perm,:]
#     object = benchmark_matmul(X_1,x_ref_1,ls=ls)
#     y_1 = object @ alpha
#     X_1=X_1.cpu()
#     y_1=y_1.cpu()
#     y_1 = y_1 + torch.randn_like(y_1)*0.1
#     data_dict_1 = {'X':X_1,'y':y_1, 'ls':ls, 'seed':seed}
#     torch.save(data_dict_1,f'uniform_probem_N={N}_eff_var={seed}.pt')
#
#
#     X_2 = torch.randn(N,d)*12**0.5
#     x_ref_2 = X_2[perm,:]
#     object = benchmark_matmul(X_2,x_ref_2,ls=ls)
#     y_2 = object @ alpha
#     X_2=X_2.cpu()
#     y_2=y_2.cpu()
#     y_2 = y_2 + torch.randn_like(y_2)*0.1
#     data_dict_2 = {'X':X_2,'y':y_2, 'ls':ls, 'seed':seed}
#     torch.save(data_dict_2,f'normal_probem_N={N}_eff_var={seed}.pt')
    #
for seed in [1,2,3]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    X_3 = torch.load('standardized_data_osm.pt')
    X_3 = X_3[:N]
    arr = np.random.randint(0,N,(5000,1))
    arr = np.unique(arr)
    alpha = torch.randn(5000,1) #Had to adjust this, else FALKON fails to converge at all
    x_ref_3 = X_3[arr[:5000],:]
    ls = torch.cdist(x_ref_3,x_ref_3).median().item()
    object = benchmark_matmul(X_3,x_ref_3,ls=ls)
    y_3 = object@alpha
    X_3=X_3.cpu()
    y_3=y_3.cpu()
    data_dict_3 = {'X':X_3,'y':y_3,'ls':ls,'seed':ls}
    # torch.save(data_dict_3,f'real_problem_N={N}_eff_var={seed}_5.pt')
    torch.save(data_dict_3,f'real_problem_N={N}_seed={seed}.pt')
    print(y_3[:100])
    #
    #
    #




