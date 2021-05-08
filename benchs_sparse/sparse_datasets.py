import torch
import numpy as np
import matplotlib.pyplot as plt

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
  f = FBM(n=n, hurst=hurst)
  X = []
  for k in range(dim):
    X.append(f.fbm())
  X = torch.tensor(X).t().contiguous()
  return X  
  
  
