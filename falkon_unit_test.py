from sklearn import datasets
import numpy as np
import torch
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import pykeops
import falkon
import time
# pykeops.clean_pykeops()          # just in case old build files are still present

if __name__ == '__main__':
    N=1000000
    X = torch.rand(N, 3).cuda()
    Y = torch.randn(N, 1).cuda()
    kernel = falkon.kernels.GaussianKernel(3.0)
    options = falkon.FalkonOptions(use_cpu=False)
    model = falkon.InCoreFalkon(kernel=kernel, penalty=1e-6, M=1000, options=options)
    start = time.time()
    model.fit(X, Y)
    end = time.time()
    print(end-start)
    preds = model.predict(X)
