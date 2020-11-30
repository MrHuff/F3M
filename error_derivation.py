from matplotlib import pyplot as plt
import numpy as np
from math import pi

def get_nodes(nr_of_nodes):
    arr = np.arange(nr_of_nodes)
    return np.cos(pi*arr/(nr_of_nodes-1))

def get_w(nr_of_nodes):
    w_list = []
    for i in range(nr_of_nodes):
        delta = 0.5 if i==0 or i==(nr_of_nodes-1) else 1.0
        w_list.append((-1**i)*delta)
    return np.array(w_list)

def calc_interpolation(x,b,nodes,w,k):
    W = w[np.newaxis,:]/(x[:,np.newaxis]-nodes[np.newaxis,:])
    W = W.sum(axis=1)
    val = w[k]/(x-nodes[k])
    return (b*val)/W

if __name__ == '__main__':
    n=100
    k = 4
    x = np.random.rand(n)*2-1
    b = np.random.randn(n)
    plt.plot(x,b,'*')
    nodes=get_nodes(k)
    w = get_w(k)
    interp_list = []
    for i in range(k):
        interp = calc_interpolation(x,b,nodes,w,i)
        # plt.plot(x,interp,'.')
        interp_list.append(interp[:,np.newaxis])
    interp_list = np.concatenate(interp_list,axis=1)
    plt.plot(x,interp_list.sum(axis=1),'r.')
    plt.plot(nodes,interp_list.sum(axis=0),'g.')
    plt.vlines(nodes,-np.max(interp_list.sum(axis=0)),np.max(interp_list.sum(axis=0)))
    plt.show()

