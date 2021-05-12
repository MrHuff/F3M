import torch
# import matplotlib.pyplot as plt
# plt.style.use('ggplot')
import falkon
from conjugate_gradient.custom_incore_falkon import InCoreFalkon_custom
from conjugate_gradient.custom_falkon import custom_Falkon
from conjugate_gradient.custom_gaussian_kernel import custom_GaussianKernel
import time
import pykeops
# pykeops.clean_pykeops()          # just in case old build files are still present

if __name__ == '__main__':
    N=10000000
    d = 3
    ls = 3.0
    penalty = 1e-5
    M = 10000
    X = torch.rand(N, d)
    Y = torch.randn(N, 1)
    kernel = custom_GaussianKernel(ls)
    options = falkon.FalkonOptions(debug=True)
    model = custom_Falkon(kernel=kernel, penalty=penalty, M=M, options=options)
    start = time.time()
    model.fit(X, Y)
    end = time.time()
    print(end-start)
    preds = model.predict(X)
    print(torch.sum((Y-preds)**2))


    kernel = falkon.kernels.GaussianKernel(ls)
    options = falkon.FalkonOptions(use_cpu=False,debug=True)
    model = falkon.Falkon(kernel=kernel, penalty=penalty, M=M, options=options)
    start = time.time()
    model.fit(X, Y)
    end = time.time()
    print(end-start)
    preds = model.predict(X)
    print(torch.sum((Y-preds)**2))

