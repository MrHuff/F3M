import torch
# import matplotlib.pyplot as plt
# plt.style.use('ggplot')
import falkon
from conjugate_gradient.custom_falkon import custom_Falkon
from conjugate_gradient.custom_gaussian_kernel import custom_GaussianKernel
from conjugate_gradient.benchmark_Gaussian_kernel import bench_GaussianKernel
from FFM_classes import *
import pykeops
# pykeops.clean_pykeops()
# pykeops.test_torch_bindings()

def calc_R2(true,pred):
    var = true.var()
    mse = torch.mean((true-pred)**2)
    r2 = 1-(mse/var)
    return r2.item()


def generate_random_problem(X,prob_size,ls):
    perm = torch.randperm(prob_size)
    x_ref = X[perm]
    true_sol = torch.randn(prob_size,1)

    tmp = benchmark_matmul(X=X,Y=x_ref,ls=ls**2)

    solve_for = tmp@true_sol

    del tmp
    torch.cuda.empty_cache()
    return solve_for.cpu()


#Fix the slack variable for keops, probably best to do custom solution for reproducibility.
if __name__ == '__main__':
    N=10000000
    d=3
    ls = 3
    penalty = 1e-5
    M = 10000
    X = torch.rand(N, d)
    Y = generate_random_problem(X,1000,ls)
    #
    kernel = custom_GaussianKernel(sigma=ls,min_points=64)
    options = falkon.FalkonOptions(use_cpu=False,debug=True)
    model = custom_Falkon(kernel=kernel, penalty=penalty, M=M, options=options)
    model.fit(X, Y)
    print(model.conjugate_gradient_time)
    preds = model.predict(X)
    r2 = calc_R2(Y,preds)
    print(r2)
    kernel = falkon.kernels.GaussianKernel(sigma=ls)
    kernel = bench_GaussianKernel(sigma=ls)
    #Homemade kernel that supposedly does the exact same thing as KeOps is also super slow compared to KeOps
    #This implies that there probably is something wrong with our code

    options = falkon.FalkonOptions(use_cpu=False,debug=True)
    model = custom_Falkon(kernel=kernel, penalty=penalty, M=M, options=options)
    model.fit(X, Y)
    print(model.conjugate_gradient_time)
    preds = model.predict(X)
    r2 = calc_R2(Y,preds)
    print(r2)

