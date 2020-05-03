import tqdm
import pandas as pd
import torch
import gpytorch
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.variational import VariationalStrategy

class GPModel(ApproximateGP):
    def __init__(self, inducing_points):
        variational_distribution = CholeskyVariationalDistribution(inducing_points.size(0))
        variational_strategy = VariationalStrategy(self, inducing_points, variational_distribution, learn_inducing_locations=True)
        super(GPModel, self).__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

def load_data(filename):
    X_train = pd.read_csv(f"{filename}.csv",header=None)
    X_train = torch.from_numpy(X_train.values).float()
    print(X_train.shape)
    return X_train

if __name__ == '__main__':
    X_train = load_data('X_train_PCA')
    Y_train = load_data('Y_train').squeeze()
    X_test = load_data('X_test_PCA')
    Y_test = load_data('Y_test').squeeze()
    if torch.cuda.is_available():
        X_train, Y_train, X_test, Y_test = X_train.cuda(), Y_train.cuda(), X_test.cuda(), Y_test.cuda()

    train_dataset = TensorDataset(X_train, Y_train)
    train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)

    test_dataset = TensorDataset(X_test, Y_test)
    test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False)


    inducing_points = X_train[:512, :]
    model = GPModel(inducing_points=inducing_points)
    likelihood = gpytorch.likelihoods.GaussianLikelihood()

    if torch.cuda.is_available():
        model = model.cuda()
        likelihood = likelihood.cuda()

    num_epochs = 50
    model.train()
    likelihood.train()
    # We use SGD here, rather than Adam. Emperically, we find that SGD is better for variational regression
    optimizer = torch.optim.Adam([
        {'params': model.parameters()},
        {'params': likelihood.parameters()},
    ], lr=0.01)

    # Our loss object. We're using the VariationalELBO
    mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=Y_train.size(0))

    epochs_iter = tqdm.tqdm(range(num_epochs), desc="Epoch")
    for i in epochs_iter:
        # Within each iteration, we will go over each minibatch of data
        minibatch_iter = tqdm.tqdm(train_loader, desc="Minibatch", leave=False)
        for x_batch, y_batch in minibatch_iter:
            optimizer.zero_grad()
            output = model(x_batch)
            loss = -mll(output, y_batch)
            minibatch_iter.set_postfix(loss=loss.item())
            loss.backward()
            optimizer.step()

    model.eval()
    likelihood.eval()
    means = torch.tensor([0.])
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            preds = model(x_batch)
            means = torch.cat([means, preds.mean.cpu()])
    means = means[1:]
    mse_loss = torch.nn.MSELoss()
    print('Test RMSE: {}'.format(mse_loss(means,Y_test.cpu()).sqrt_().item()))
    #Test RMSE: 0.24515990912914276
