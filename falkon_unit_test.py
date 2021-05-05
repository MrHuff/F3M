from sklearn import datasets
import numpy as np
import torch
import matplotlib.pyplot as plt
plt.style.use('ggplot')

import falkon


if __name__ == '__main__':
    X, Y = datasets.load_boston(return_X_y=True)
    num_train = int(X.shape[0] * 0.8)
    num_test = X.shape[0] - num_train
    shuffle_idx = np.arange(X.shape[0])
    np.random.shuffle(shuffle_idx)
    train_idx = shuffle_idx[:num_train]
    test_idx = shuffle_idx[num_train:]

    Xtrain, Ytrain = X[train_idx], Y[train_idx]
    Xtest, Ytest = X[test_idx], Y[test_idx]
    # convert numpy -> pytorch
    Xtrain = torch.from_numpy(Xtrain)
    Xtest = torch.from_numpy(Xtest)
    Ytrain = torch.from_numpy(Ytrain)
    Ytest = torch.from_numpy(Ytest)

    # z-score normalization
    train_mean = Xtrain.mean(0, keepdim=True)
    train_std = Xtrain.std(0, keepdim=True)
    Xtrain -= train_mean
    Xtrain /= train_std
    Xtest -= train_mean
    Xtest /= train_std

    options = falkon.FalkonOptions()
    kernel = falkon.kernels.GaussianKernel(sigma=5)
    flk = falkon.Falkon(kernel=kernel, penalty=1e-5, M=Xtrain.shape[0], options=options)

    flk.fit(Xtrain, Ytrain)

    train_pred = flk.predict(Xtrain).reshape(-1, )
    test_pred = flk.predict(Xtest).reshape(-1, )


    def rmse(true, pred):
        return torch.sqrt(torch.mean((true.reshape(-1, 1) - pred.reshape(-1, 1)) ** 2))


    print("Training RMSE: %.3f" % (rmse(train_pred, Ytrain)))
    print("Test RMSE: %.3f" % (rmse(test_pred, Ytest)))

    fig, ax = plt.subplots()
    ax.hist(Ytest, alpha=0.7, label="True")
    ax.hist(test_pred, alpha=0.7, label="Pred")
    ax.legend(loc="best")