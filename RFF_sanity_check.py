import torch
import time
class RFF():
    def __init__(self,m,d):
        self.m = m
        self.d = d
        self.coeff  = (2/d)**0.5
        self.w = torch.randn(m,d)
        self.b = torch.rand(1,d)

    def transform(self,X):
        return self.coeff*torch.cos(X@self.w+self.b.expand(X.shape[0],-1))

if __name__ == '__main__':
    block_1 = torch.randn(100,3)
    block_2 = torch.randn(100,3)
    b_1 = torch.rand(100,1)
    b_2 = torch.rand(100,1)

    rff = RFF(3,15)

    s = time.time()
    cat_block = torch.cat([block_1,block_2],dim=0)
    cat_RFF = rff.transform(cat_block)
    cat_im = cat_RFF.t()@torch.cat([b_1,b_2],dim=0)
    cat_res = cat_RFF@cat_im
    e = time.time()
    print(e-s)


    block_1_RFF = rff.transform(block_1)
    block_2_RFF = rff.transform(block_2)
    block_im =block_1_RFF.t()@b_1+block_2_RFF.t()@b_2
        # [block_1_RFF.t()@b_1,block_2_RFF.t()@b_2]
    # print(cat_im)
    # print(block_1_RFF.t()@b_1+block_2_RFF.t()@b_2)
    block_res = []
    for el in [block_1_RFF,block_2_RFF]:
        res_tmp = el@block_im
        # for el_2 in block_im:
        #     res_tmp+=el@el_2

        block_res.append(res_tmp)
    block_res = torch.cat(block_res,dim=0)
    e_2 = time.time()
    print(e_2-e)

    print(cat_res)
    print(block_res)







