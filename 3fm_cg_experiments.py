from FFM_classes import *
from conjugate_gradient.conjugate_gradient import ConjugateGradientSolver,KernelLinearSolver


if __name__ == '__main__':
    N = 1000000
    eff_var = 1
    problem_set = torch.load(f'uniform_probem_N={N}_eff_var={eff_var}.pt')
    X = problem_set['X']
    X = X.cuda()
    y = problem_set['y']
    y=y.cuda()
    ls = problem_set['ls']

    ffm_obj = FFM(X=X,ls=ls,min_points=1000,nr_of_interpolation=64,eff_var_limit=0.3,var_compression=True,small_field_points=64)
    # ffm_obj = benchmark_matmul(X, ls=ls)
    def linop(b):
        return ffm_obj@b
    bindings = 'torch'
    sol = KernelLinearSolver(FFM_obj=ffm_obj,binding=bindings,x=X,b=y,precond=True,K=('gaussian',3,1,ls),alpha=1e-3)
    # sol = ConjugateGradientSolver(binding=bindings,linop=linop,eps=1e-6,b=y)
