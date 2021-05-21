import numpy as np
from math import pi
from gpytorch.kernels import RBFKernel,Kernel
import torch
from sklearn.preprocessing import minmax_scale
import itertools
import pandas as pd
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

    if nr_of_nodes==0:
        approx_res = torch.zeros_like(res)
        rel_error = torch.norm(res-approx_res)/torch.norm(res)
        abs_error = torch.norm(res-approx_res)
        if torch.isnan(rel_error):
            rel_error=abs_error
        return rel_error,abs_error
    else:
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
        gathered = []

        for el in node_idx:
            gathered.append([nodes[j] for j in el])
        gathered = torch.tensor(gathered)
        cheb_data  =(edge/2)*gathered +edge/2
        interp_X,interp_b_X = get_interpolation_list(nodes,w,X,center_x,factor,b,node_idx)
        interp_Y,interp_b_Y = get_interpolation_list(nodes,w,Y,center_y,factor,b,node_idx)

        y_cheb = cheb_data + center_y - center_x  #cheb_data....
        x_cheb = cheb_data
        if y_cheb.dim()==1:
            y_cheb=y_cheb.unsqueeze(0)
            x_cheb=x_cheb.unsqueeze(0)
        with torch.no_grad():
            mid_ker =  rbf(x_cheb,y_cheb).evaluate()
            mid_res =mid_ker@interp_b_Y.sum(dim=0)
            approx_res = interp_X @ mid_res  #incorrect, not symmetric!
            kernel_approx =interp_X@(mid_ker@ interp_Y.t())
        rel_error = torch.norm(res-approx_res)/torch.norm(res)
        abs_error = torch.norm(res-approx_res)
        print('Relative error: ',rel_error)
        print('abs error: ',abs_error)
        with torch.no_grad():
            real_kernel = rbf(X,Y).evaluate()
        print('rel ERROR kernel: ', torch.norm(kernel_approx-real_kernel)/torch.norm(real_kernel))
        if torch.isnan(rel_error):
            rel_error=abs_error
        return rel_error,abs_error

def run_experiment(d,eff_far_field,nr_intpol):
    n = 5000
    X = torch.rand(n,d)
    Y = X+2
    b = torch.randn(n,1)
    ls = (0.5/eff_far_field)**0.5
    rel_err,abs_error = interpolation_experiment(X,Y,b,ls,nr_intpol)
    return rel_err,abs_error,eff_far_field,nr_intpol**d

if __name__ == '__main__':
    # d = 3
    # ls = 1.0
    # nr_intpol = 4
    d_list = [3,4,5]
    ls_list = [1e-3,1e-2,1e-1,1,2,3,4,5,6,7,8,9,10,100,1000]
    node_list = [0,1,2,3,4,5]
    columns = ['d','eff_far_field','nodes','rel_error','abs_error']
    data_list = []
    for el in itertools.product(d_list,ls_list,node_list):
        rel_err,abs_error,eff_far_field,nodes = run_experiment(*el)
        data_list.append([el[0],eff_far_field,nodes,rel_err.item(),abs_error.item()])
    df = pd.DataFrame(data_list,columns=columns)
    df.to_csv("rbf_experiment_error.csv")









