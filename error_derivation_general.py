import numpy as np
from math import pi
from gpytorch.kernels import RBFKernel,Kernel
import torch
from sklearn.preprocessing import minmax_scale
import itertools

def get_edge_etc(x):
    edge = (x.max() - x.min())*1.01
    factor = 2./edge
    min_val,_=x.min(0)
    center =  min_val + edge/2 +1e-6
    return edge,factor,center

def get_nodes(nr_of_nodes):
    arr = torch.arange(nr_of_nodes)
    if nr_of_nodes>1:
        return torch.cos(pi*arr/(nr_of_nodes-1))
    else:
        return torch.cos(pi*arr/(nr_of_nodes))

def get_w(nr_of_nodes):
    w_list = []
    for i in range(nr_of_nodes):
        delta = 0.5 if i==0 or i==(nr_of_nodes-1) else 1.0
        w_list.append(((-1)**i)*delta)
    return torch.tensor(w_list)

def calc_l_k(x,nodes,w):
    bools = nodes[np.newaxis,:] == x[:,np.newaxis]
    W = w[np.newaxis,:]/(x[:,np.newaxis]-nodes[np.newaxis,:])
    W = W.sum(axis=1)
    indicator = torch.sum(bools,dim=1)>0
    W[indicator]=1.0
    W=1./W
    return W,indicator

def get_LK(x,nodes,w):
    d = x.shape[1]
    LK=[]
    BOOLEAN = []
    for i in range(d):
        lk,boolean= calc_l_k(x[:,i],nodes,w)
        LK.append(lk)
        BOOLEAN.append(boolean)
    return torch.stack(LK,dim=1),torch.stack(BOOLEAN,dim=1)

def calc_interpolation(X,k,nodes,w,W,indicator):
    val = (w[k]/(X-nodes[k]))*W
    val[indicator.squeeze()]=1.0
    return val

def calc_interpolation_d(X,nodes,w,W,indicator,k_list):
    d = X.shape[1]
    res = torch.ones(X.shape[0])
    for j in range(d):
        res*=calc_interpolation(X[:,j],k_list[j],nodes,w,W[:,j],indicator[:,j])
    return res

def get_interpolation_list(nodes,w,y,center_y,factor_y,b,combs):
    dat_y = factor_y*(y-center_y)
    W,indicator = get_LK(dat_y,nodes,w)
    interp_list = []
    for k_list in combs:
        interp = calc_interpolation_d(dat_y,nodes,w,W,indicator,k_list) #weird scaling issue...
        interp_list.append(interp)
    interp_list = torch.stack(interp_list,dim=1)
    summed = interp_list*b
    return interp_list,summed

def interpolation_experiment(X,Y,b,ls,nr_of_nodes):

    rbf = RBFKernel()
    ls = torch.tensor(ls)
    rbf.lengthscale=ls
    with torch.no_grad():
        res = rbf(X,Y)@b
        res = res.squeeze()
    d = X.shape[1]
    edge_x,_,center_x = get_edge_etc(X)
    edge_y,_,center_y = get_edge_etc(Y)
    edge = max(edge_x,edge_y)
    print('edge',edge)
    factor = 2./edge

    node_list = [i for i in range(nr_of_nodes)]
    cart_prod_list = [node_list for i in range(d)]
    node_idx = list(itertools.product(*cart_prod_list))
    nodes = get_nodes(nr_of_nodes)
    w = get_w(nr_of_nodes)
    cheb_data  =(edge/2)*nodes[np.array(node_idx)].float()+edge/2

    interp_X,interp_b_X = get_interpolation_list(nodes,w,X,center_x,factor,b,node_idx)
    interp_Y,interp_b_Y = get_interpolation_list(nodes,w,Y,center_y,factor,b,node_idx)

    y_cheb = cheb_data + center_y - center_x  #cheb_data....
    x_cheb = cheb_data

    with torch.no_grad():
        mid_ker =  rbf(x_cheb,y_cheb).evaluate()
        print('midker ',mid_ker)
        mid_res =mid_ker@interp_b_Y.sum(dim=0)
        approx_res = interp_X @ mid_res  #incorrect, not symmetric!
        kernel_approx =interp_X@(mid_ker@ interp_Y.t())
    print(res[:10])
    print(approx_res[:10])
    print('Relative error: ',torch.norm(res-approx_res)/torch.norm(res))
    print(kernel_approx[:10,:])
    with torch.no_grad():
        real_kernel = rbf(X,Y).evaluate()
    print('rel ERROR kernel: ', torch.norm(kernel_approx-real_kernel)/torch.norm(real_kernel))
    print(real_kernel[:10,:].sum(1))

if __name__ == '__main__':
    n = 1000
    d = 3
    X = torch.rand(n,d)
    Y = X
    b = torch.randn(n,1)

    interpolation_experiment(X,Y,b,10.0,4)





