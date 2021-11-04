import numpy as np
# from thundersvm import SVC
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.datasets import make_blobs, make_circles, make_moons
from FFM_classes import *
import time
tol = 0.01 # error tolerance
eps = 0.01 # alpha tolerance




def objective_function(alphas, target, kernel, X_train):
    """Returns the SVM objective function based in the input model defined by:
    `alphas`: vector of Lagrange multipliers
    `target`: vector of class labels (-1 or 1) for training data
    `kernel`: kernel function
    `X_train`: training data for model."""



    return np.sum(alphas) - 0.5 * np.sum((target[:, None] * target[None, :]) * kernel(X_train, X_train) * (alphas[:, None] * alphas[None, :]))

def gpu_objective_function(alphas,target,kernel_op):

    alp = torch.from_numpy(alphas).float().to('cuda:0')
    tar = torch.from_numpy(target).float().to('cuda:0')
    prod = (alp*tar).unsqueeze(-1)
    return   (torch.sum(alp)-torch.sum((kernel_op@prod) * prod)).cpu().numpy()

def gpu_decision_function(alphas, target, kernel_op, b):
    """Applies the SVM decision function to the input feature vectors in `x_test`."""
    alp = torch.from_numpy(alphas).float().to('cuda:0')
    tar = torch.from_numpy(target).float().to('cuda:0')
    prod = (alp*tar).unsqueeze(-1)
    result = (kernel_op@(prod)).squeeze() - b
    return result.cpu().numpy()

def decision_function(alphas, target, kernel, X_train, x_test, b):
    """Applies the SVM decision function to the input feature vectors in `x_test`."""

    result = (alphas * target) @ kernel(X_train, x_test) - b
    return result


def take_step(i1, i2, model):
    # Skip if chosen alphas are the same
    if i1 == i2:
        return 0, model

    alph1 = model.alphas[i1]
    alph2 = model.alphas[i2]
    y1 = model.y[i1]
    y2 = model.y[i2]
    E1 = model.errors[i1]
    E2 = model.errors[i2]
    s = y1 * y2

    # Compute L & H, the bounds on new possible alpha values
    if (y1 != y2):
        L = max(0, alph2 - alph1)
        H = min(model.C, model.C + alph2 - alph1)
    elif (y1 == y2):
        L = max(0, alph1 + alph2 - model.C)
        H = min(model.C, alph1 + alph2)
    if (L == H):
        return 0, model

    # Compute kernel & 2nd derivative eta
    k11 = model.kernel(model.X[i1], model.X[i1])
    k12 = model.kernel(model.X[i1], model.X[i2])
    k22 = model.kernel(model.X[i2], model.X[i2])
    eta = 2 * k12 - k11 - k22

    # Compute new alpha 2 (a2) if eta is negative
    if (eta < 0):
        a2 = alph2 - y2 * (E1 - E2) / eta
        # Clip a2 based on bounds L & H
        if L < a2 < H:
            a2 = a2
        elif (a2 <= L):
            a2 = L
        elif (a2 >= H):
            a2 = H

    # If eta is non-negative, move new a2 to bound with greater objective function value
    else:
        alphas_adj = model.alphas.copy()
        alphas_adj[i2] = L
        # objective function output with a2 = L
        Lobj = gpu_objective_function(alphas_adj, model.y,kernel_op=model.kernel_op_train)
        alphas_adj[i2] = H
        # objective function output with a2 = H
        Hobj = gpu_objective_function(alphas_adj, model.y,kernel_op=model.kernel_op_train)
        if Lobj > (Hobj + eps):
            a2 = L
        elif Lobj < (Hobj - eps):
            a2 = H
        else:
            a2 = alph2

    # Push a2 to 0 or C if very close
    if a2 < 1e-8:
        a2 = 0.0
    elif a2 > (model.C - 1e-8):
        a2 = model.C

    # If examples can't be optimized within epsilon (eps), skip this pair
    if (np.abs(a2 - alph2) < eps * (a2 + alph2 + eps)):
        return 0, model

    # Calculate new alpha 1 (a1)
    a1 = alph1 + s * (alph2 - a2)

    # Update threshold b to reflect newly calculated alphas
    # Calculate both possible thresholds
    b1 = E1 + y1 * (a1 - alph1) * k11 + y2 * (a2 - alph2) * k12 + model.b
    b2 = E2 + y1 * (a1 - alph1) * k12 + y2 * (a2 - alph2) * k22 + model.b

    # Set new threshold based on if a1 or a2 is bound by L and/or H
    if 0 < a1 and a1 < C:
        b_new = b1
    elif 0 < a2 and a2 < C:
        b_new = b2
    # Average thresholds if both are bound
    else:
        b_new = (b1 + b2) * 0.5

    # Update model object with new alphas & threshold
    model.alphas[i1] = a1
    model.alphas[i2] = a2

    # Update error cache
    # Error cache for optimized alphas is set to 0 if they're unbound
    for index, alph in zip([i1, i2], [a1, a2]):
        if 0.0 < alph < model.C:
            model.errors[index] = 0.0

    # Set non-optimized errors based on equation 12.11 in Platt's book
    non_opt = [n for n in range(model.m) if (n != i1 and n != i2)]
    model.errors[non_opt] = model.errors[non_opt] + \
                            y1 * (a1 - alph1) * model.kernel(model.X[i1], model.X[non_opt]) + \
                            y2 * (a2 - alph2) * model.kernel(model.X[i2], model.X[non_opt]) + model.b - b_new

    # Update model threshold
    model.b = b_new

    return 1, model


def examine_example(i2, model):
    y2 = model.y[i2]
    alph2 = model.alphas[i2]
    E2 = model.errors[i2]
    r2 = E2 * y2

    # Proceed if error is within specified tolerance (tol)
    if ((r2 < -tol and alph2 < model.C) or (r2 > tol and alph2 > 0)):

        if len(model.alphas[(model.alphas != 0) & (model.alphas != model.C)]) > 1:
            # Use 2nd choice heuristic is choose max difference in error
            if model.errors[i2] > 0:
                i1 = np.argmin(model.errors)
            elif model.errors[i2] <= 0:
                i1 = np.argmax(model.errors)
            step_result, model = take_step(i1, i2, model)
            if step_result:
                return 1, model

        # Loop through non-zero and non-C alphas, starting at a random point
        for i1 in np.roll(np.where((model.alphas != 0) & (model.alphas != model.C))[0],
                          np.random.choice(np.arange(model.m))):
            step_result, model = take_step(i1, i2, model)
            if step_result:
                return 1, model

        # loop through all alphas, starting at a random point
        for i1 in np.roll(np.arange(model.m), np.random.choice(np.arange(model.m))):
            step_result, model = take_step(i1, i2, model)
            if step_result:
                return 1, model

    return 0, model


def gaussian_kernel(x, y, sigma=1):
    """Returns the gaussian similarity of arrays `x` and `y` with
    kernel width parameter `sigma` (set to 1 by default)."""

    if np.ndim(x) == 1 and np.ndim(y) == 1:
        result = np.exp(- (np.linalg.norm(x - y, 2)) ** 2 / (2 * sigma ** 2))
    elif (np.ndim(x) > 1 and np.ndim(y) == 1) or (np.ndim(x) == 1 and np.ndim(y) > 1):
        result = np.exp(- (np.linalg.norm(x - y, 2, axis=1) ** 2) / (2 * sigma ** 2))
    elif np.ndim(x) > 1 and np.ndim(y) > 1:
        result = np.exp(- (np.linalg.norm(x[:, np.newaxis] - y[np.newaxis, :], 2, axis=2) ** 2) / (2 * sigma ** 2))
    return result

class SMOModel:
    """Container object for the model used for sequential minimal optimization."""

    def __init__(self, X, y, C, kernel, alphas, b, errors,kernel_op_type='keops'):
        self.X = X  # training data vector
        self.y = y  # class label vector
        self.C = C  # regularization parameter
        self.kernel = kernel  # kernel function
        self.alphas = alphas  # lagrange multiplier vector
        self.b = b  # scalar bias term
        self.errors = errors  # error cache
        self._obj = []  # record of objective function value
        self.m = len(self.X)  # store size of training set
        self.tensor_X = torch.from_numpy(X)
        if kernel_op_type=='keops':
            self.kernel_op_train = benchmark_matmul(self.tensor_X, ls=1.0, device='cuda:0')
        else:
            self.kernel_op_train = FFM(X=self.tensor_X, ls=1.0, min_points=64, nr_of_interpolation=64,
                                              eff_var_limit=0.9, var_compression=True,
                                              device='cuda:0', small_field_points=64)
        self.kernel_op_test = self.kernel_op_train


def train(model):
    numChanged = 0
    examineAll = 1
    count = 0
    # while (numChanged > 0) or (examineAll):
    for i in range(20):
        print(f'num it: {count}')
        numChanged = 0
        if examineAll:
            # loop over all training examples
            for i in range(model.alphas.shape[0]):
                examine_result, model = examine_example(i, model)
                numChanged += examine_result
                if examine_result:
                    obj_result = gpu_objective_function(model.alphas, model.y, model.kernel_op_train)
                    model._obj.append(obj_result)
        else:
            # loop over examples where alphas are not already at their limits
            for i in np.where((model.alphas != 0) & (model.alphas != model.C))[0]:
                examine_result, model = examine_example(i, model)
                numChanged += examine_result
                if examine_result:
                    obj_result = gpu_objective_function(model.alphas, model.y, model.kernel_op_train)
                    model._obj.append(obj_result)
        if examineAll == 1:
            examineAll = 0
        elif numChanged == 0:
            examineAll = 1
        count+=1
    return model

def sample_spherical(npoints, ndim=3):
    vec = np.random.randn(npoints,ndim)
    vec /= np.linalg.norm(vec, axis=1)[:,np.newaxis]
    return vec

def generate_data(n,d):
    n_class_1=n//2
    ball_1 = sample_spherical(n_class_1,d) * np.random.rand(n_class_1,1)*0.1

    # fig = plt.figure()
    # ax = Axes3D(fig)
    # ax.scatter(ball_1[:,0],ball_1[:,1],ball_1[:,2],c = 'b', marker='o')
    # plt.show()

    pos = np.ones(n_class_1)
    ball_2 = sample_spherical(n_class_1,d) * (np.random.rand(n_class_1,1)*0.2+0.8)
    neg = -np.ones(n_class_1)
    dat = np.concatenate([ball_1,ball_2],axis=0)
    labels = np.concatenate([pos,neg],axis=0)
    return dat,labels


mode='keops'
X,y = generate_data(1000000,3)
m = len(X)
initial_alphas = np.zeros(m)
initial_b = 0.0
C = 1.0


model = SMOModel(X, y, C, lambda x, y: gaussian_kernel(x, y, sigma=1.0),
                 initial_alphas, initial_b, np.zeros(m),mode)
start= time.time()
initial_error = np.sign(gpu_decision_function(model.alphas, model.y, model.kernel_op_test, model.b)) - model.y
model.errors = initial_error
output = train(model)
later_error = np.sign(gpu_decision_function(model.alphas, model.y, model.kernel_op_test, model.b)) - model.y
print(later_error.sum())
end = time.time()
time = end-start

file1 = open("times.txt", "w")
L = [f"{mode} : {time} \n"]
file1.writelines(L)
file1.close()

# clf.fit(x_train,y_train)
# y_predict=clf.predict(x_test)
# score=clf.score(x_test,y_test)



