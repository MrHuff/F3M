from F3M_src.FFM_classes import *
import pykeops
# pykeops.clean_pykeops()
import time

n=1000000 #Nr of observations
device = "cuda:0" #device
dim=3 #dims, stick to <5
X = torch.randn(n,dim).float().to(device) #gene
ls=1.0
b = torch.randn(n,1).float().to(device) #weights

bench_1 = benchmark_matmul(X,ls=ls,device=device) #get some references
bench_2 = keops_matmul(X,ls=ls,device=device) #get some references


s = time.time()
bench_1@b
t = time.time()
print(t-s)


s = time.time()
bench_2@b
t = time.time()
print(t-s)

