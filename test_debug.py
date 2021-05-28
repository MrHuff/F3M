import torch
#
from FFM_classes import *
import time
from run_obj import *


# import pykeops
# pykeops.clean_pykeops()
class Container(torch.nn.Module):
    def __init__(self, my_values):
        super().__init__()
        for key in my_values:
            setattr(self, key, my_values[key])

if __name__ == '__main__':
    for seed in [1,2,3]:
        N=1000000000
        problem_set = torch.load(f'real_problem_N={N}_seed={seed}.pt')
        X = problem_set['X']
        Y = problem_set['y']
        # print(Y[:1000])
        # my_values = {
        #     'X':X.float(),
        #     'b':Y.float()
        #
        # }
        # container = torch.jit.script(Container(my_values))
        # container.save("../faulty_data_2.pt")

        Y = problem_set['y']
        ls = problem_set['ls']
        x_ref = X[:5000]
        nodes = 16
        obj_test = FFM(X=X,ls=ls,min_points=5000,eff_var_limit=2.0,var_compression=True,small_field_points=nodes,nr_of_interpolation=nodes)
        start= time.time()
        res = obj_test@Y
        end = time.time()
        print(end-start)

        bmark_2 = benchmark_matmul(X=x_ref,Y=X,ls=ls)
        res_2 = bmark_2@Y

        print(torch.norm(res[:5000]-res_2)/torch.norm(res_2))
        del obj_test, bmark_2, res,res_2,Y,X,ls
        torch.cuda.empty_cache()


