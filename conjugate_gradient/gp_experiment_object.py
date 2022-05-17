import hyperopt
from hyperopt import STATUS_OK,Trials
import gpytorch
import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold,KFold
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA,KernelPCA
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from FFM_classes import *
from conjugate_gradient.conjugate_gradient import ConjugateGradientSolver,PreConditionedConjugateGradientSolver,nystrom_preconditioner,get_GP_likelihood
from gpytorch.kernels.rbf_kernel import RBFKernel

import tqdm
def sq_dist( x1, x2):
    adjustment = x1.mean(-2, keepdim=True)
    x1 = x1 - adjustment
    x2 = x2 - adjustment  # x1 and x2 should be identical in all dims except -2 at this point

    # Compute squared distance matrix using quadratic expansion
    x1_norm = x1.pow(2).sum(dim=-1, keepdim=True)
    x1_pad = torch.ones_like(x1_norm)
    x2_norm = x2.pow(2).sum(dim=-1, keepdim=True)
    x2_pad = torch.ones_like(x2_norm)
    x1_ = torch.cat([-2.0 * x1, x1_norm, x1_pad], dim=-1)
    x2_ = torch.cat([x2, x2_pad, x2_norm], dim=-1)
    res = x1_.matmul(x2_.transpose(-2, -1))
    # Zero out negative values
    res.clamp_min_(0)

    # res  = torch.cdist(x1,x2,p=2)
    # Zero out negative values
    # res.clamp_min_(0)
    return res


def covar_dist( x1, x2):
    return sq_dist(x1, x2).sqrt()


def get_median_ls(X, Y=None):
    if X.shape[0] > 5000:
        idx = torch.randperm(X.shape[0])[:5000]
        X = X[idx, :]
    with torch.no_grad():
        if Y is None:
            d = covar_dist(x1=X, x2=X)
        else:
            if Y.shape[0] > 5000:
                idx = torch.randperm(Y.shape[0])[:5000]
                Y = Y[idx, :]
            d = covar_dist(x1=X, x2=Y)
        ret = torch.sqrt(torch.median(d[d >= 0]))  # print this value, should be increasing with d
        if ret.item() == 0:
            ret = torch.tensor(1.0)
        return ret


class GPRegressionModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood,num_dims):
        super(GPRegressionModel, self).__init__(train_x, train_y, likelihood)

        # SKI requires a grid size hyperparameter. This util can help with that
        grid_size = gpytorch.utils.grid.choose_grid_size(train_x)

        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.GridInterpolationKernel(
                gpytorch.kernels.RBFKernel(), grid_size=grid_size, num_dims=num_dims
            )
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)



class StratifiedKFold3(KFold):
    def split(self, X, y, groups=None,stratify=False):
        s = super().split(X, y, groups)
        fold_indices=[]
        for train_indxs, test_indxs in s:
            y_train = y[train_indxs]
            if stratify:
                train_indxs, cv_indxs = train_test_split(train_indxs,stratify=y_train, test_size=(1 / (self.n_splits - 1)))
            else:
                train_indxs, cv_indxs = train_test_split(train_indxs, test_size=(1 / (self.n_splits)))

            # yield train_indxs, cv_indxs, test_indxs
            fold_indices.append((train_indxs, cv_indxs, test_indxs))
        return fold_indices
class experiment_object_gp():
    def __init__(self,job_parameters):
        self.fold = job_parameters['fold']
        self.total_folds = job_parameters['total_folds']
        self.ds_name = job_parameters['ds_name']
        self.model_string = job_parameters['model_string']
        self.seed = job_parameters['seed']
        self.pca_comps = job_parameters['pca_comps']
        self.device = job_parameters['device']
        self.nr_of_its = job_parameters['nr_of_its']
        self.do_pca = job_parameters['do_pca']
        self.use_precond = job_parameters['use_precond']

    def preprocess_data(self):
        X = np.load(self.ds_name + '/X.npy', allow_pickle=True)
        y = np.load(self.ds_name + '/y.npy', allow_pickle=True)
        indices = np.arange(y.shape[0])
        tr_ind, val_ind, test_ind = StratifiedKFold3(n_splits=self.total_folds, shuffle=True, random_state=self.seed).split(indices, y)[self.fold]
        s = StandardScaler()
        s_2 = StandardScaler()
        # if self.do_pca:
        #     pca = PCA(n_components=self.pca_comps, svd_solver='full')
        #     X = pca.fit_transform(X)
        #     # test_X = pca.transform(test_X)
        #     # val_X = pca.transform(val_X)

        tr_X = s.fit_transform(X[tr_ind, :])
        val_X = s.fit_transform(X[val_ind, :])
        test_X = s.transform(X[test_ind,:])

        tr_Y = s_2.fit_transform( y[tr_ind].reshape(-1, 1)).squeeze()
        val_Y = s_2.fit_transform( y[val_ind].reshape(-1, 1)).squeeze()
        test_Y = s_2.transform(y[test_ind].reshape(-1, 1)).squeeze()
        # tr_Y = y[tr_ind]
        # val_Y =y[val_ind]
        # test_Y =y[test_ind]

        if self.do_pca:
            pca = PCA(n_components=self.pca_comps, svd_solver='full')
            tr_X = pca.fit_transform(tr_X)
            test_X = pca.transform(test_X)
            val_X = pca.transform(val_X)

        self.X_tr = torch.from_numpy(tr_X).float().to(self.device)
        self.X_val = torch.from_numpy(val_X).float().to(self.device)
        self.X_tst = torch.from_numpy(test_X).float().to(self.device)

        self.Y_tr = torch.from_numpy(tr_Y).float().to(self.device)
        self.Y_val = torch.from_numpy(val_Y).float().to(self.device)
        self.Y_tst = torch.from_numpy(test_Y).float().to(self.device)

    def rsme(self,pred,Y):
        mse =torch.nn.MSELoss()
        loss = mse(pred.squeeze(),Y.squeeze())
        return loss.item()**0.5

    def ski_validate(self,mode='val'):
        self.model.eval()
        self.likelihood.eval()
        if mode=='val':
            X = self.X_val
            y = self.Y_val
        else:
            X = self.X_tst
            y = self.Y_tst
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            observed_pred = self.likelihood(self.model(X))
            mean = observed_pred.mean
        test_rsme = self.rsme(mean,y)
        return test_rsme
    def run_ski(self):
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
        self.model = GPRegressionModel(self.X_tr, self.Y_tr, self.likelihood, self.X_tr.shape[1])
        self.model = self.model.to(self.device)
        self.model.train()
        self.likelihood.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters

        # "Loss" for GPs - the marginal log likelihood
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)

        best=np.inf
        for i in tqdm.tqdm(range(self.nr_of_its)):
            self.model.train()
            self.likelihood.train()
            optimizer.zero_grad()
            output = self.model(self.X_tr)
            loss = -mll(output, self.Y_tr)
            print(loss.item())
            loss.backward()
            optimizer.step()
            if i%5==0:
                val_error = self.ski_validate('val')
                print(val_error.item())
                if val_error.item()<best:
                    best=val_error.item()
        test_rsme = self.ski_validate('test')
        return test_rsme



    def run_F3M(self,parameters):
        base_ls = get_median_ls(self.X_tr)
        base_ls = base_ls*parameters['ls_scale']
        l=parameters['lamb']
        ffm_obj = FFM(X=self.X_tr,ls=base_ls,min_points=64,nr_of_interpolation=64,eff_var_limit=0.5,var_compression=True,small_field_points=64)
        # ffm_obj = benchmark_matmul(X=self.X_tr,ls=base_ls) #ill-conditioned...
        bindings = 'torch'
        if self.use_precond:
            pre_cond_kernel = RBFKernel()
            pre_cond_kernel._set_lengthscale(base_ls)
            pre_cond_kernel = pre_cond_kernel.cuda()
            precond = nystrom_preconditioner(self.X_tr, 1.0,lamb=l)
            precond.calculate_constants(pre_cond_kernel)
            alpha,alpha_mat,beta_mat = PreConditionedConjugateGradientSolver(preconditioner=precond, binding=bindings, kmvm_object=ffm_obj,
                                                        lamb=l, eps=1e-2, b=self.Y_tr.unsqueeze(-1))
        # Preconditioner needed...
        else:
            alpha,alpha_mat,beta_mat = ConjugateGradientSolver(binding=bindings,kmvm_object=ffm_obj,eps=1e-3,b=self.Y_tr.unsqueeze(-1),lamb=l)
        if alpha_mat.shape[1]>1:
            alpha_mat, beta_mat = alpha_mat[:,1:],beta_mat[:,1:]
            alpha=alpha[:,0]
            NLL = get_GP_likelihood(alpha,self.Y_tr,alpha_mat,beta_mat)

        pred = ffm_obj.forward(self.X_tst,self.X_tr,b=alpha)
        test_rsme = self.rsme(pred,self.Y_tst)

        pred = ffm_obj.forward(self.X_val,self.X_tr,b=alpha)
        val_rsme = self.rsme(pred,self.Y_val)

        return {'status':STATUS_OK,'val_rsme':val_rsme,'test_rsme':test_rsme}


