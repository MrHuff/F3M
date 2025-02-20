from matplotlib import pyplot as plt
from math import pi
from gpytorch.kernels import RBFKernel
import torch
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

if __name__ == '__main__':
    rbf = RBFKernel()
    dist = 0
    n=100
    if dist==0:
        d = torch.distributions.normal.Normal(0,1.0)
    elif dist==1:
        d = torch.distributions.uniform.Uniform(0,1)
    x = d.sample((n,1))
    # x,_ = x.sort(0)
    x_plot,y_plot = np.meshgrid(x.numpy(),x.numpy())
    # m=np.median((x_plot-y_plot)**2)
    m=1.0
    print(m)
    rbf._set_lengthscale(torch.tensor(m).sqrt())
    with torch.no_grad():
        k_xy = rbf(x,x).evaluate().numpy()
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(X=x_plot,Y=y_plot,Z=k_xy,cmap=cm.coolwarm,
                        linewidth=0, antialiased=False)
    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.show()





