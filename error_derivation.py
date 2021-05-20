import numpy as np
from math import pi
from gpytorch.kernels import RBFKernel,Kernel
import torch
from sklearn.preprocessing import minmax_scale

def get_nodes(nr_of_nodes):
    arr = np.arange(nr_of_nodes)
    if nr_of_nodes>1:
        return np.cos(pi*arr/(nr_of_nodes-1))
    else:
        return np.cos(pi*arr/(nr_of_nodes))

def get_w(nr_of_nodes):
    w_list = []
    for i in range(nr_of_nodes):
        delta = 0.5 if i==0 or i==(nr_of_nodes-1) else 1.0
        w_list.append(((-1)**i)*delta)
    return np.array(w_list)

def calc_W(x,nodes,w):
    bools = nodes[np.newaxis,:] == x[:,np.newaxis]
    W = w[np.newaxis,:]/(x[:,np.newaxis]-nodes[np.newaxis,:])
    W = W.sum(axis=1)
    indicator = np.sum(bools,axis=1)>0
    W[indicator]=1.0
    W=1./W
    return W,indicator
    # return W

def calc_interpolation(X,k,nodes,w,W,indicator):
    val = (w[k]/(X-nodes[k]))*W

    val[indicator]=1.0

    return val

def get_ls(X):
    c=Kernel()
    distance = c.covar_dist(X,X)
    m = torch.median(distance)**0.5
    return m

def get_smallest_distance_from_edge(x,edge):
    min_x = x.min()
    min_dist = np.min(0.5-np.abs((x-min_x)/edge-0.5))*0.5*edge
    return min_dist


def interpolation_scale_invariant(x,k):
    # x = minmax_scale(x)
    edge = (np.max(x) - np.min(x))
    factor = 2./edge
    center =  np.min(x) + edge/2 +1e-6
    # ls = torch.median(torch.cdist(torch.from_numpy(x).float().unsqueeze(-1),torch.from_numpy(x).float().unsqueeze(-1)))
    # ls = torch.tensor(1e-6)#torch.median(torch.cdist(torch.from_numpy(x).float().unsqueeze(-1),torch.from_numpy(x).float().unsqueeze(-1))).sqrt()
    ls = torch.tensor(1.0)
    print("ls ", ls)
    rbf.lengthscale=ls
    with torch.no_grad():
        res = rbf(torch.from_numpy(x).float(),torch.from_numpy(x).float())@torch.from_numpy(b).float()
    nodes=get_nodes(k)
    w = get_w(k)
    dat = factor*(x-center)

    W,indicator = calc_W(dat,nodes,w)
    interp_list = []
    # idx = np.argsort(x)
    print(dat.max())
    print(dat.min())
    for i in range(k):
        #factor*(x-center)
        interp = calc_interpolation(dat,i,nodes,w,W,indicator) #weird scaling issue...
        # plt.plot(x,interp,'.')
        interp_list.append(interp[:,np.newaxis])
    interp_list = np.concatenate(interp_list,axis=1)
    summed = interp_list*b[:,np.newaxis]
    with torch.no_grad():
        x_nodes = (edge/2)*torch.from_numpy(nodes).float()+edge/2
        mid_ker =  rbf(x_nodes,x_nodes).evaluate()
        print('midker ',mid_ker)
        mid_res =mid_ker@torch.from_numpy(summed.sum(axis=0)).float()
        approx_res = torch.from_numpy(interp_list).float() @ mid_res
        kernel_approx = torch.from_numpy(interp_list).float()@(mid_ker@ torch.from_numpy(interp_list).float().t())
    print(res[:10])
    print(approx_res[:10])
    print('Relative error: ',((res-approx_res)/res).abs().mean())
    # print(kernel_approx[:10,:])
    # with torch.no_grad():
    #     print(rbf(torch.from_numpy(x).float(),torch.from_numpy(x).float()).evaluate()[:10,:])

def get_edge_etc(x):
    edge = (np.max(x) - np.min(x))
    factor = 2./edge
    center =  np.min(x) + edge/2 +1e-6
    return edge,factor,center

def get_interpolation_list(nodes,w,y,center_y,factor_y,b):
    dat_y = factor_y*(y-center_y)
    W,indicator = calc_W(dat_y,nodes,w)
    interp_list = []
    for i in range(k):
        interp = calc_interpolation(dat_y,i,nodes,w,W,indicator) #weird scaling issue...
        interp_list.append(interp[:,np.newaxis])
    interp_list = np.concatenate(interp_list,axis=1)
    summed = interp_list*b[:,np.newaxis]
    return interp_list,summed

def interpolation_xy(x,y,k,ls): #Key is equivariance such that the same "edge" can be applied. after that it's only center distance.
    edge_x,factor_x,center_x = get_edge_etc(x)
    edge_y,factor_y,center_y = get_edge_etc(y)
    edge = max(edge_x,edge_y)
    print('edge',edge)
    factor = 2./edge
    ls = torch.tensor(ls)
    rbf.lengthscale=ls
    print(rbf)
    with torch.no_grad():
        res = rbf(torch.from_numpy(x).float(),torch.from_numpy(y).float())@torch.from_numpy(b).float()
    nodes=get_nodes(k)
    w = get_w(k)
    interp_list_y,summed_y = get_interpolation_list(nodes,w,y,center_y,factor,b)
    interp_list_x,summed_x = get_interpolation_list(nodes,w,x,center_x,factor,b)
    cheb_data  =(edge/2)*torch.from_numpy(nodes).float()+edge/2
    test = rbf(torch.from_numpy(np.array([center_x])),torch.from_numpy(np.array([center_y]))).evaluate()


    with torch.no_grad():
        y_cheb = cheb_data + center_y - center_x  #cheb_data....
        mid_ker =  rbf(cheb_data,y_cheb).evaluate()
        print('midker ',mid_ker)
        mid_res =mid_ker@torch.from_numpy(summed_y.sum(axis=0)).float()
        approx_res = torch.from_numpy(interp_list_x).float() @ mid_res  #incorrect, not symmetric!
        # approx_res = (torch.ones_like(approx_res)*test*np.sum(b)).sq
        kernel_approx = torch.from_numpy(interp_list_x).float()@(mid_ker@ torch.from_numpy(interp_list_y).float().t())
    print(res[:10])
    print(approx_res.squeeze()[:10])
    print('Relative error: ',torch.norm(res-approx_res)/torch.norm(res))
    print(kernel_approx[:10,:])
    with torch.no_grad():
        real_kernel = rbf(torch.from_numpy(x).float(),torch.from_numpy(y).float()).evaluate()

    print('rel ERROR kernel: ', torch.norm(kernel_approx-real_kernel)/torch.norm(real_kernel))
    print(real_kernel[:10,:].sum(1))
if __name__ == '__main__':
    rbf = RBFKernel()

    n=1000
    k =100
    x = np.random.rand(n)
    y = x
    b = np.random.randn(n)

    #Interpolation does not work relatively well when we are very far away, i.e. the exponent is very small -> close to 0 results.


    # interpolation_scale_invariant(x,k)
    # interpolation_scale_invariant(x*100,k)

    #low (effective) variance diagonal entries can be interpolated for faster speed. (only for X times X)
    #Failure mode is when b has a lot of weight specifically where things are in the "far field".

    interpolation_xy(x,y,k,ls=1.0)
    #hmm interpolation in "far field or large lengthscales generally quite bad...
