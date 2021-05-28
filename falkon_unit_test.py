import torch
# import matplotlib.pyplot as plt
# plt.style.use('ggplot')
import falkon
from conjugate_gradient.custom_falkon import custom_Falkon
from conjugate_gradient.custom_gaussian_kernel import custom_GaussianKernel
from FFM_classes import *
import pykeops
# pykeops.clean_pykeops()
# pykeops.test_torch_bindings()

def calc_R2(true,pred):
    var = true.var()
    mse = torch.mean((true-pred)**2)
    r2 = 1-(mse/var)
    return r2.item()

#Fix the slack variable for keops, probably best to do custom solution for reproducibility.
if __name__ == '__main__':
    N=100000000
    M=10000

    eff_var=0.1 # 0.1 - 1e-3, 1 - 1e-3, 10, 1e-2?, 1e-2 is already acceptable... Unif
    # eff_var=10 # 0.1 - 1e-3, 1 - 1e-3, 10, 1e-2?, 1e-2 is already acceptable... Norm
    penalty = 1e-2 #does a little better, seems like 3F-M might need a little more penalty...
    problem_set = torch.load(f'small_real_problem_N={N}_eff_var={eff_var}_5.pt')
    X = problem_set['X']
    Y = problem_set['y']
    ls = problem_set['ls']
    #
    # OK PICK YOUR POISON, EITHER ONE IS SLOW OR ONE IS FAST DEPENDING ON TRANSPOSE, MIGHT ALMOST SWITCH TAG IN BETWEEN
    # WHAT THE FUCK IS GOING ON????
    nr_of_interpolation_nodes = 27
    kernel = custom_GaussianKernel(sigma=ls,min_points=250,var_compression=True,interpolation_nr=nr_of_interpolation_nodes,eff_var_limit=0.5)
    options = falkon.FalkonOptions(use_cpu=False,debug=True)
    model = custom_Falkon(kernel=kernel, penalty=penalty, M=M, options=options)
    model.fit(X, Y)
    print(model.conjugate_gradient_time)
    preds = model.predict(X)
    r2 = calc_R2(Y,preds)
    print(r2)
    kernel = falkon.kernels.GaussianKernel(sigma=ls)
    options = falkon.FalkonOptions(use_cpu=False,debug=True)
    model = custom_Falkon(kernel=kernel, penalty=penalty, M=M, options=options)
    model.fit(X, Y)
    print(model.conjugate_gradient_time)
    preds = model.predict(X)
    r2 = calc_R2(Y,preds)
    print(r2)

