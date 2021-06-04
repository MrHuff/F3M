import torch
import numpy as np
import matplotlib.pyplot as plt


def UnitBox(X):
    X = X - X.min(dim=0, keepdims=True)[0]
    X = X / X.max(dim=0, keepdims=True)[0]
    return X
def PlotDataSave2(X,Y, max_npoints = np.inf,name='fig_exotic.png'):
    n, dim = X.shape
    m = min(n,max_npoints)
    if dim==2:
        n = X.shape[0]
        plt.figure(figsize=(12,10))
        plt.plot(X[:m,0], X[:m,1], '.', markersize=2)
        plt.plot(Y[:m,0], Y[:m,1], '.',color='r', markersize=2)
        plt.axis('equal')
        plt.savefig(name,bbox_inches = 'tight',pad_inches = 0)
    elif dim==3:
        ax = plt.figure(figsize=(16,10)).add_subplot(projection='3d')
        X = X.numpy()
        ax.plot(X[:m,0], X[:m,1], X[:m,2], '.', markersize=.5)
    else:
        raise ValueError("not implemented")
def PlotDataSave(X, max_npoints = np.inf,name='fig_exotic.png'):
    n, dim = X.shape
    m = min(n,max_npoints)
    if dim==2:
        n = X.shape[0]
        plt.figure(figsize=(12,10))
        plt.plot(X[:m,0], X[:m,1], '.', markersize=2)
        plt.axis('equal')
        plt.savefig(name,bbox_inches = 'tight',pad_inches = 0)
    elif dim==3:
        ax = plt.figure(figsize=(16,10)).add_subplot(projection='3d')
        X = X.numpy()
        ax.plot(X[:m,0], X[:m,1], X[:m,2], '.', markersize=.5)
    else:
        raise ValueError("not implemented")

def PlotData(X, max_npoints = np.inf):
    n, dim = X.shape
    m = min(n,max_npoints)
    if dim==2:
        n = X.shape[0]
        plt.figure(figsize=(16,10))
        plt.plot(X[:m,0], X[:m,1], '.', markersize=.5)
        plt.axis('equal')
    elif dim==3:
        ax = plt.figure(figsize=(16,10)).add_subplot(projection='3d')
        X = X.numpy()
        ax.plot(X[:m,0], X[:m,1], X[:m,2], '.', markersize=.5)
    else:
        raise ValueError("not implemented")


def MultiScaleClusterData(dim,n_branches,r):
    sigma = n = 1
    X = np.zeros((1,dim))
    for n_br in n_branches:
        n *= n_br
        X = np.repeat(X,n_br,axis=0) + sigma * np.random.randn(n,dim)
        sigma *= r
    return torch.from_numpy(X)

        
        
def MaternClusterData(dim,lam,mu,r):
    Np = np.random.poisson(lam)
    C = (1+2*r) * np.random.rand(Np,dim)
    Nd = np.random.poisson(mu,Np)
    N = sum(Nd)
    X = np.zeros((N,dim))
    if dim==2:
        rho = r * np.sqrt(np.random.rand(N))
        theta = 2 * np.pi * np.random.rand(N)
        X[:,0] = rho * np.cos(theta)
        X[:,1] = rho * np.sin(theta)
    elif dim==3:
        rho = r * np.random.rand(N) ** (1/3)
        phi = 2 * np.pi * np.random.rand(N)
        theta = np.arccos(2 * np.random.rand(N) - 1)
        X[:,0] = rho * np.sin(theta) * np.cos(phi)
        X[:,1] = rho * np.sin(theta) * np.sin(phi)
        X[:,2] = rho * np.cos(theta)
    else:
        raise ValueError("not implemented")
    X += np.repeat(C,Nd,axis=0)
    X = X[np.all((X>0)&(X<1),axis=1),:]
    return torch.from_numpy(X)



def FBMData(dim, n, hurst):
    from fbm import FBM
    neff = min(10000000,n)
    f = FBM(n=neff-1, hurst=hurst)
    X = []
    for k in range(dim):
        nrem = n
        Xk = np.array([])
        start_val = 0
        while nrem>0:
            Xtmp = f.fbm()[:neff]
            Xk = np.concatenate((Xk,Xtmp-Xtmp[0]+start_val))
            start_val = 2*Xk[-1]-Xk[-2]
            nrem -= neff
        X.append(Xk)
    X = torch.tensor(X).t().contiguous()
    return X  
  
  








    
def Uniform2D1e6():
    title = "Uniform_D2_N1e6"
    X = torch.rand(1000000,2)
    return X, title

def Uniform2D1e7():
    title = "Uniform_D2_N1e7"
    X = torch.rand(10000000,2)
    return X, title

def Uniform2D1e8():
    title = "Uniform_D2_N1e8"
    X = torch.rand(100000000,2)
    return X, title

def Uniform2D1e9():
    title = "Uniform_D2_N1e9"
    X = torch.rand(1000000000,2)
    return X, title



def Uniform3D1e6():
    title = "Uniform_D3_N1e6"
    X = torch.rand(1000000,3)
    return X, title

def Uniform3D1e7():
    title = "Uniform_D3_N1e7"
    X = torch.rand(10000000,3)
    return X, title

def Uniform3D1e8():
    title = "Uniform_D3_N1e8"
    X = torch.rand(100000000,3)
    return X, title

def Uniform3D1e9():
    title = "Uniform_D3_N1e9"
    X = torch.rand(1000000000,3)
    return X, title




def ClusteredDataset2D1e6():
    title = "Clustered_Dataset_D2_N1e6"
    X = MultiScaleClusterData(2,[100,100,100],.03)
    return UnitBox(X), title

def ClusteredDataset2D1e7():
    title = "Clustered_Dataset_D2_N1e7"
    X = MultiScaleClusterData(2,[10,100,100,100],.03)
    return UnitBox(X), title

def ClusteredDataset2D1e8():
    title = "Clustered_Dataset_D2_N1e8"
    X = MultiScaleClusterData(2,[100,100,100,100],.03)
    return UnitBox(X), title

def ClusteredDataset2D1e9():
    title = "Clustered_Dataset_D2_N1e9"
    X = MultiScaleClusterData(2,[10,100,100,100,100],.03)
    return UnitBox(X), title



def ClusteredDataset3D1e6():
    title = "Clustered_Dataset_D3_N1e6"
    X = MultiScaleClusterData(3,[100,100,100],.03)
    return UnitBox(X), title

def ClusteredDataset3D1e7():
    title = "Clustered_Dataset_D3_N1e7"
    X = MultiScaleClusterData(3,[10,100,100,100],.03)
    return UnitBox(X), title

def ClusteredDataset3D1e8():
    title = "Clustered_Dataset_D3_N1e8"
    X = MultiScaleClusterData(3,[100,100,100,100],.03)
    return UnitBox(X), title

def ClusteredDataset3D1e9():
    title = "Clustered_Dataset_D3_N1e9"
    X = MultiScaleClusterData(3,[10,100,100,100,100],.03)
    return UnitBox(X), title








def FBMDataset2D1e6():
    title = "Fractional_Brownian_Motion_Dataset_D2_N1e6"
    X = FBMData(2,1000000,0.75)
    return UnitBox(X), title
    
def FBMDataset2D1e7():
    title = "Fractional_Brownian_Motion_Dataset_D2_N1e7"
    X = FBMData(2,10000000,0.75)
    return UnitBox(X), title
    
def FBMDataset2D1e8():
    title = "Fractional_Brownian_Motion_Dataset_D2_N1e8"
    X = FBMData(2,100000000,0.75)
    return UnitBox(X), title
    
def FBMDataset2D1e9():
    title = "Fractional_Brownian_Motion_Dataset_D2_N1e9"
    X = FBMData(2,1000000000,0.75)
    return UnitBox(X), title
    
    
def FBMDataset3D1e6():
    title = "Fractional_Brownian_Motion_Dataset_D3_N1e6"
    X = FBMData(3,1000000,0.75)
    return UnitBox(X), title
    
def FBMDataset3D1e7():
    title = "Fractional_Brownian_Motion_Dataset_D3_N1e7"
    X = FBMData(3,10000000,0.75)
    return UnitBox(X), title
    
def FBMDataset3D1e8():
    title = "Fractional_Brownian_Motion_Dataset_D3_N1e8"
    X = FBMData(3,100000000,0.75)
    return UnitBox(X), title
    
def FBMDataset3D1e9():
    title = "Fractional_Brownian_Motion_Dataset_D3_N1e9"
    X = FBMData(3,1000000000,0.75)
    return UnitBox(X), title
    
    
    

def BMDataset2D1e6():
    title = "Brownian_Motion_Dataset_D2_N1e6"
    X = FBMData(2,1000000,0.5)
    return UnitBox(X), title
    
def BMDataset2D1e7():
    title = "Brownian_Motion_Dataset_D2_N1e7"
    X = FBMData(2,10000000,0.5)
    return UnitBox(X), title
    
def BMDataset2D1e8():
    title = "Brownian_Motion_Dataset_D2_N1e8"
    X = FBMData(2,100000000,0.5)
    return UnitBox(X), title
    
def BMDataset2D1e9():
    title = "Brownian_Motion_Dataset_D2_N1e9"
    X = FBMData(2,1000000000,0.5)
    return UnitBox(X), title
    
    
def BMDataset3D1e6():
    title = "Brownian_Motion_Dataset_D3_N1e6"
    X = FBMData(3,1000000,0.5)
    return UnitBox(X), title
    
def BMDataset3D1e7():
    title = "Brownian_Motion_Dataset_D3_N1e7"
    X = FBMData(3,10000000,0.5)
    return UnitBox(X), title
    
def BMDataset3D1e8():
    title = "Brownian_Motion_Dataset_D3_N1e8"
    X = FBMData(3,100000000,0.5)
    return UnitBox(X), title
    
def BMDataset3D1e9():
    title = "Brownian_Motion_Dataset_D3_N1e9"
    X = FBMData(3,1000000000,0.5)
    return UnitBox(X), title
    
    
    

