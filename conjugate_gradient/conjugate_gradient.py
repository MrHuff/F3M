import numpy as np

from pykeops.common.utils import get_tools
import torch
import tqdm
# Some advance operations defined at user level use in fact other reductions.
def preprocess(reduction_op, formula2):
    reduction_op = reduction_op

    if (
        reduction_op == "SumSoftMaxWeight" or reduction_op == "SoftMax"
    ):  # SoftMax is just old naming for SumSoftMaxWeight
        # SumSoftMaxWeight relies on KeOps Max_SumShiftExpWeight reduction, with a custom finalize
        reduction_op_internal = "Max_SumShiftExpWeight"
        # we concatenate the 2nd formula (g) with a constant 1, so that we get sum_j exp(m_i-f_ij) g_ij and sum_j exp(m_i-f_ij) together
        formula2 = "Concat(IntCst(1)," + formula2 + ")"
    elif reduction_op == "LogSumExp":
        # LogSumExp relies also on Max_SumShiftExp or Max_SumShiftExpWeight reductions, with a custom finalize
        if formula2:
            # here we want to compute a log-sum-exp with weights: log(sum_j(exp(f_ij)g_ij))
            reduction_op_internal = "Max_SumShiftExpWeight"
        else:
            # here we want to compute a usual log-sum-exp: log(sum_j(exp(f_ij)))
            reduction_op_internal = "Max_SumShiftExp"
    else:
        reduction_op_internal = reduction_op

    return reduction_op_internal, formula2


def postprocess(out, binding, reduction_op, nout, opt_arg, dtype):
    tools = get_tools(binding)
    # Post-processing of the output:
    if reduction_op == "SumSoftMaxWeight" or reduction_op == "SoftMax":
        # we compute sum_j exp(f_ij) g_ij / sum_j exp(f_ij) from sum_j exp(m_i-f_ij) [1,g_ij]
        out = out[..., 2:] / out[..., 1][..., None]
    elif reduction_op == "ArgMin" or reduction_op == "ArgMax":
        # outputs are encoded as floats but correspond to indices, so we cast to integers
        out = tools.long(out)
    elif (
        reduction_op == "Min_ArgMin"
        or reduction_op == "MinArgMin"
        or reduction_op == "Max_ArgMax"
        or reduction_op == "MaxArgMax"
    ):
        # output is one array of size N x 2D, giving min and argmin value for each dimension.
        # We convert to one array of floats of size NxD giving mins, and one array of size NxD giving argmins (casted to integers)
        shape_out = out.shape
        tmp = tools.view(out, shape_out[:-1] + (2, -1))
        vals = tmp[..., 0, :]
        indices = tmp[..., 1, :]
        out = (vals, tools.long(indices))
    elif reduction_op == "KMin":
        # output is of size N x KD giving K minimal values for each dim. We convert to array of size N x K x D
        shape_out = out.shape
        out = tools.view(out, shape_out[:-1] + (opt_arg, -1))
        if out.shape[-1] == 1:
            out = out.squeeze(-1)
    elif reduction_op == "ArgKMin":
        # output is of size N x KD giving K minimal values for each dim. We convert to array of size N x K x D
        # and cast to integers
        shape_out = out.shape
        out = tools.view(tools.long(out), shape_out[:-1] + (opt_arg, -1))
        if out.shape[-1] == 1:
            out = out.squeeze(-1)
    elif reduction_op == "KMin_ArgKMin" or reduction_op == "KMinArgKMin":
        # output is of size N x 2KD giving K min and argmin for each dim. We convert to 2 arrays of size N x K x D
        # and cast to integers the second array
        shape_out = out.shape
        out = tools.view(out, shape_out[:-1] + (opt_arg, 2, -1))
        out = (out[..., 0, :], tools.long(out[..., 1, :]))
        if out[0].shape[-1] == 1:
            out = (out[0].squeeze(-1), out[1].squeeze(-1))
    elif reduction_op == "LogSumExp":
        # finalize the log-sum-exp computation as m + log(s)
        if out.shape[-1] == 2:  # means (m,s) with m scalar and s scalar
            out = (out[..., 0] + tools.log(out[..., 1]))[..., None]
        else:  # here out.shape[-1]>2, means (m,s) with m scalar and s vectorial
            out = out[..., 0][..., None] + tools.log(out[..., 1:])
    return out



def subsample_inducing_points(X,m_fac=1.0):
    sqrt_n = X.shape[0]**0.5
    M = int(round(m_fac*sqrt_n))
    idx = torch.randperm(X.shape[0])[:M]
    X_M = torch.clone(X[idx])
    return X_M

class nystrom_preconditioner:
    def __init__(self,X,m_fac=1.0,lamb=1e-2):
        self.X = X
        self.x_m = subsample_inducing_points(self.X,m_fac)
        self.lamb=lamb
    def calculate_constants(self,memory_kernel):
        self.k_X_x_m = memory_kernel(self.X,self.x_m).evaluate()
        self.small_in_mem = memory_kernel(self.x_m).evaluate() + self.k_X_x_m.t()@self.k_X_x_m/self.lamb
        # self.L = torch.linalg.cholesky(small_in_mem)
        self.inside_inverse = torch.inverse(self.small_in_mem)
        # self.inside_inverse = torch.cholesky_inverse(L)

    def apply_predcond(self,ffm_obj,b):
        right = ffm_obj.forward(self.x_m,self.X,b)/self.lamb
        mid = self.inside_inverse@right
        # mid = torch.linalg.solve(self.small_in_mem,right)
        left = ffm_obj.forward(self.X,self.x_m,mid)
        P_inv_vec = (b-left)/self.lamb
        return P_inv_vec

def PreConditionedConjugateGradientSolver(binding,preconditioner,kmvm_object, b,lamb=1e-3, eps=1e-6,max_its=1000):
    def linop(b):
        return kmvm_object@b+lamb*b
    tools = get_tools(binding)
    delta = eps
    u = 0
    r = tools.copy(b)
    z = preconditioner.apply_predcond(ffm_obj=kmvm_object,b=r)
    d = tools.copy(z)
    rel_err = b.abs().sum().item()
    nz2 = (z * r).mean(dim=0,keepdim=True)
    alpha_matrix = []
    beta_matrix  = []
    for i in tqdm.tqdm(range(max_its)):
        error = torch.abs(r).sum() / rel_err
        print(error.mean().item())
        if error.mean().item() < delta:
            print(error)
            break
        v = linop(d)
        alp = (nz2) / (d * v).mean(dim=0,keepdim=True)
        u += alp * d
        r -= alp * v

        z = preconditioner.apply_predcond(ffm_obj=kmvm_object, b=r)
        nz2new = (z * r).mean(dim=0,keepdim=True)  # mean square residual... #can do z as well for some reason...
        d = z + (nz2new / nz2) * d
        nz2 = torch.clone(nz2new)
    alpha_matrix = torch.cat(alpha_matrix,dim=0)
    beta_matrix = torch.cat(beta_matrix,dim=0)
    return u,alpha_matrix,beta_matrix

def get_GP_likelihood(alpha,y,alpha_mat,beta_mat):
    t_mat = build_tridiagonalization_matrices(alpha_mat,beta_mat)
    log_kXX=calculate_log_kxx(t_mat)
    NLL =log_kXX - torch.sum(alpha*y).item()
    return NLL
def calculate_log_kxx(T_list):
    a = torch.linalg.eigvalsh(T_list)
    val, ind = a.max(1)
    return val.log().mean().item()


def build_tridiagonalization_matrices(alphas,betas):
    d = alphas.shape[1]
    T_matrices = []
    for i in range(d):
        # alpha_minus_1 = torch.cat([torch.zeros_like(alphas[0,:]),alphas[1:]])
        # beta_minus_1 = torch.cat([torch.zeros_like(betas[0,:]),betas[1:]])
        alpha_minus_1 = torch.clone(alphas[:,i])
        alpha_minus_1[0]=0.0
        beta_minus_1 = torch.clone(betas[:,i])
        beta_minus_1[0]=0.0
        diag = 1/alphas[:,i] + torch.nan_to_num(beta_minus_1/alpha_minus_1,nan=0.0)
        off_diag = betas[1:,i]**0.5/alphas[1:,i]
        s= torch.diag(off_diag,1)
        T= torch.diag(diag)+s+s.t()
        T_matrices.append(T)
    T_matrices = torch.stack(T_matrices,dim=0)
    return T_matrices

def ConjugateGradientSolver(binding, kmvm_object, b,lamb=1e-2 ,eps=1e-6,max_its=1000):
    # Conjugate gradient algorithm to solve linear system of the form
    # Ma=b where linop is a linear operation corresponding
    # to a symmetric and positive definite matrix
    def linop(b):
        return kmvm_object@b+lamb*b
    tools = get_tools(binding)
    delta = eps
    a = 0
    r = tools.copy(b)
    nr2 = (r ** 2).mean(dim=0,keepdim=True)

    p = tools.copy(r)
    k = 0
    rel_err = b.abs().sum(dim=0,keepdim=True)
    alpha_matrix = []
    beta_matrix  = []
    for i in tqdm.tqdm(range(max_its)):
        Mp = linop(p)
        alp = nr2 / (p * Mp).mean(dim=0,keepdim=True)
        a += alp * p
        r -= alp * Mp
        error = torch.abs(r).sum(dim=0,keepdim=True)/rel_err
        print(error.mean().item())
        if error.mean().item() < delta:
            print(error)
            break
        nr2new = (r ** 2).mean(dim=0,keepdim=True)
        beta = (nr2new / nr2)
        p = r +  beta * p
        nr2 = nr2new
        alpha_matrix.append(alp)
        beta_matrix.append(beta)
        k += 1
    alpha_matrix = torch.cat(alpha_matrix,dim=0)
    beta_matrix = torch.cat(beta_matrix,dim=0)
    return a,alpha_matrix,beta_matrix
