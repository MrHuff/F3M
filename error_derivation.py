from matplotlib import pyplot as plt
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

    return W,indicator
    # return W

def calc_interpolation(X,k,nodes,w,W,indicator):
    val = (w[k]/(X-nodes[k]))/W

    val[indicator]=1.0

    return val

def get_ls(X):
    c=Kernel()
    distance = c.covar_dist(X,X)
    m = torch.median(distance)**0.5
    return m

def interpolation_scale_invariant(x,k):
    # x = minmax_scale(x)
    edge = (np.max(x) - np.min(x))
    factor = 2./edge
    center =  np.min(x) + edge/2 +1e-6
    # ls = torch.median(torch.cdist(torch.from_numpy(x).float().unsqueeze(-1),torch.from_numpy(x).float().unsqueeze(-1)))
    ls = torch.tensor(1e-6)#torch.median(torch.cdist(torch.from_numpy(x).float().unsqueeze(-1),torch.from_numpy(x).float().unsqueeze(-1))).sqrt()
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
    print(kernel_approx[:10,:])
    with torch.no_grad():
        print(rbf(torch.from_numpy(x).float(),torch.from_numpy(x).float()).evaluate()[:10,:])

if __name__ == '__main__':
    rbf = RBFKernel()

    n=100
    k =4
    x = np.random.randn(n)
    #
    b = np.random.randn(n)

    interpolation_scale_invariant(x,k)
    # interpolation_scale_invariant(x*100,k)



    # plt.plot(x,b,'*')
    # plt.plot(x,interp_list.sum(axis=1),'r.')
    # plt.show()
    # plt.plot(nodes,interp_list.sum(axis=0),'g.')
    # plt.vlines(nodes,-np.max(interp_list.sum(axis=0)),np.max(interp_list.sum(axis=0)))
    # plt.show()
    # plt.vlines(nodes,-np.max(interp_list.sum(axis=0)),np.max(interp_list.sum(axis=0)))
    # plt.plot(nodes,interp_list.transpose()[:,1],'.')
    # plt.show()

