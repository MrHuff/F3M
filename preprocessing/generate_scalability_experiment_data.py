import torch
import os
seed=1
def generate_folder_of_subblocks(par_fac,n,d,x_sp):
    torch.manual_seed(seed)
    X = torch.empty(n, d).uniform_(0, (1 * 12) ** 0.5)
    b = torch.empty(n, 1).normal_(0, 1)
    y_chunks = torch.chunk(X, par_fac, dim=0)
    x_chunks = torch.chunk(X, x_sp, dim=0)
    b_chunks = torch.chunk(b, par_fac, dim=0)
    dat_list = []
    for y_sub, b_sub in zip(y_chunks, b_chunks):
        for x_sub in x_chunks:
            dat_list.append((x_sub,y_sub,b_sub))
    torch.save(dat_list,f'uniform_{d}_{n}_{par_fac}.pt')

if __name__ == '__main__':
    d=3
    for n in [1000000,10000000,100000000,1000000000]:
        for par_fac,x_sp in zip([1,2,4],[1,2,2]):
            generate_folder_of_subblocks(par_fac,n,d,x_sp)



