import torch
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import falkon
from conjugate_gradient.custom_incore_falkon import InCoreFalkon_custom
from conjugate_gradient.custom_gaussian_kernel import custom_GaussianKernel
import time
import pykeops
# pykeops.clean_pykeops()          # just in case old build files are still present

if __name__ == '__main__':
    N=1000000
    X = torch.rand(N, 3).cuda()
    Y = torch.randn(N, 1).cuda()
    kernel = custom_GaussianKernel(3.0)
    options = falkon.FalkonOptions(use_cpu=False,debug=True)
    model = InCoreFalkon_custom(kernel=kernel, penalty=1, M=1000, options=options)
    start = time.time()
    model.fit(X, Y)
    end = time.time()
    print(end-start)
    preds = model.predict(X)
    print(torch.sum((Y-preds)**2))
