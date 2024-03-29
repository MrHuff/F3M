import os.path

# # from FFM_classes import *
# from conjugate_gradient.conjugate_gradient import ConjugateGradientSolver,\
#     PreConditionedConjugateGradientSolver,nystrom_preconditioner,subsample_inducing_points,build_tridiagonalization_matrices
# import torch
# from gpytorch.kernels.rbf_kernel import RBFKernel
# from conjugate_gradient.gp_experiment_object import experiment_object_gp
import pickle
#first step is to just cheese it with KeOps
#if you have more time you can try to implement it on your own
def generate_jobs():
    job_dict = {
        'fold':0,
        'total_folds':3,
        'ds_name':'',
        'model_string':'',
        'seed':42,
        'pca_comps':3,
        'device':'cuda:0',
        'nr_of_its':100,
        'do_pca':True,
        'use_precond':False
    }
    #Figure out the woodbury inversion formula for preconditioning business try the Nyström and possibly the pivoted Cholesky stuff
    #.
    jname = 'job_f3m'
    if not os.path.exists(jname):
        os.makedirs(jname)
    for model,its in zip(['f3m'],[100]):
        a,b = [False, True, True], ['3D_spatial', 'Song','Buzz',]
        # a,b = [False, True, True,True], ['3D_spatial', 'Song','Buzz', 'household']
        for do_pca,job in zip(a,b):#3D_spatial, household
            for f in [0,1,2]:
                job_dict['fold'] = f
                job_dict['do_pca']=do_pca
                job_dict['ds_name']=job
                job_dict['model_string']=model
                job_dict['nr_of_its'] = its
                with open(f'{jname}/{model}_dataset={job}_fold={f}.pickle', 'wb') as handle:
                    pickle.dump(job_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    # N = 1000000
    # eff_var = 1
    # # #Normal data is plain ill-conditioned
    # # # You can almost solve any ill-conditioned problem with strong regularization or PC
    # problem_set = torch.load(f'normal_probem_N={N}_eff_var={eff_var}.pt')
    # X = problem_set['X']
    # X = X.cuda()
    # y = problem_set['y']
    # y=y.cuda()
    # y_2 = torch.randn(y.shape[0],10).cuda()
    # y = torch.cat([y,y_2],dim=1)
    # ls = problem_set['ls']#10
    # ffm_obj = FFM(X=X,ls=ls,min_points=64,nr_of_interpolation=64,eff_var_limit=0.5,var_compression=True,small_field_points=64)
    # # ffm_obj = benchmark_matmul(X=X,ls=ls) #ill-conditioned...
    #
    # bindings = 'torch'
    # sol,alpha_mat,beta_mat = ConjugateGradientSolver(binding=bindings,kmvm_object=ffm_obj,lamb=1e-1,eps=1e-2,b=y,max_its=100)
    # T_matrices = build_tridiagonalization_matrices(alpha_mat,beta_mat)
    # print(T_matrices.shape)

    # pre_cond_kernel = RBFKernel()
    # pre_cond_kernel._set_lengthscale(ls)
    # pre_cond_kernel = pre_cond_kernel.cuda()

    """
    sanity check
    """
    # pre_cond_kernel = pre_cond_kernel.cuda()
    # x_m = subsample_inducing_points(X,0.1)
    # b = torch.ones_like(x_m)
    # res_1 = pre_cond_kernel(X,x_m).evaluate()@b
    # res_2 = ffm_obj.forward(X,x_m,b)

    # precond = nystrom_preconditioner(X,0.5)
    # precond.calculate_constants(pre_cond_kernel)
    # sol = PreConditionedConjugateGradientSolver(preconditioner=precond,binding=bindings,kmvm_object=ffm_obj,lamb=1e-1,eps=1e-2,b=y)
    # output = ffm_obj.grad(y)

    generate_jobs()




