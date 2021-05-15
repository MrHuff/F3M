import torch
# import matplotlib.pyplot as plt
# plt.style.use('ggplot')
import falkon
from conjugate_gradient.custom_incore_falkon import InCoreFalkon_custom
from conjugate_gradient.custom_falkon import custom_Falkon
from conjugate_gradient.custom_gaussian_kernel import custom_GaussianKernel
from conjugate_gradient.benchmark_Gaussian_kernel import bench_GaussianKernel
from conjugate_gradient.FALKON_V100_fix.kernels import GaussianKernelV100
import time
import pykeops
# pykeops.test_torch_bindings()
# pykeops.clean_pykeops()

"""
MainProcess.MainThread::[Calcuating Preconditioner of size 100000]
Preconditioner will run on 1 GPUs
--MainProcess.MainThread::[Kernel]
--MainProcess.MainThread::[Kernel] complete in 40.603s
--MainProcess.MainThread::[Cholesky 1]
Using parallel POTRF
--MainProcess.MainThread::[Cholesky 1] complete in 35.445s
--MainProcess.MainThread::[Copy triangular]
--MainProcess.MainThread::[Copy triangular] complete in 8.664s
--MainProcess.MainThread::[LAUUM(CUDA)]
--MainProcess.MainThread::[LAUUM(CUDA)] complete in 48.307s
--MainProcess.MainThread::[Cholesky 2]
Using parallel POTRF
--MainProcess.MainThread::[Cholesky 2] complete in 37.133s
MainProcess.MainThread::[Calcuating Preconditioner of size 100000] complete in 170.154s
MainProcess.MainThread::[Computing Falkon iterations]
Optimizer will run on 1 GPUs
MainProcess.MainThread::[Computing Falkon iterations] complete in 1740.634s
1913.5735561847687
tensor(9.9996e+08)
MainProcess.MainThread::[Calcuating Preconditioner of size 100000]
Preconditioner will run on 1 GPUs
--MainProcess.MainThread::[Kernel]
--MainProcess.MainThread::[Kernel] complete in 18.417s
--MainProcess.MainThread::[Cholesky 1]
Using parallel POTRF
--MainProcess.MainThread::[Cholesky 1] complete in 35.256s
--MainProcess.MainThread::[Copy triangular]
--MainProcess.MainThread::[Copy triangular] complete in 8.860s
--MainProcess.MainThread::[LAUUM(CUDA)]
--MainProcess.MainThread::[LAUUM(CUDA)] complete in 47.634s
--MainProcess.MainThread::[Cholesky 2]
Using parallel POTRF
--MainProcess.MainThread::[Cholesky 2] complete in 37.167s
MainProcess.MainThread::[Calcuating Preconditioner of size 100000] complete in 147.336s
MainProcess.MainThread::[Computing Falkon iterations]
Optimizer will run on 1 GPUs
cudaSafeCall() failed at /data/greyostrich/not-backed-up/nvme00/rhu/miniconda3/envs/new_nnenv/lib/python3.8/site-packages/pykeops/cmake_scripts/script_keops_formula/../../keops/core/mapreduce/GpuConv1D.cu:432 : out of memory

Process finished with exit code 255


"""

#Fix the slack variable for keops, probably best to do custom solution for reproducibility.
if __name__ == '__main__':
    N=1000000000
    d = 3
    ls = 3.0
    penalty = 1e-5
    M = 10000
    X = torch.rand(N, d)
    Y = torch.randn(N, 1)
    kernel = custom_GaussianKernel(ls,min_points=64)
    options = falkon.FalkonOptions(use_cpu=False,debug=True)
    model = custom_Falkon(kernel=kernel, penalty=penalty, M=M, options=options)
    start = time.time()
    model.fit(X, Y)
    end = time.time()
    print(end-start)
    preds = model.predict(X)
    print(torch.sum((Y-preds)**2))


    kernel = bench_GaussianKernel(ls)
    options = falkon.FalkonOptions(use_cpu=False,debug=True)
    model = custom_Falkon(kernel=kernel, penalty=penalty, M=M, options=options)
    start = time.time()
    model.fit(X, Y)
    end = time.time()
    print(end-start)
    preds = model.predict(X)
    print(torch.sum((Y-preds)**2))

