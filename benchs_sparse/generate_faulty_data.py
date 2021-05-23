# import torch
# from sparse_datasets import *
# from FFMbench import FFMbench, PlotBench
# import matplotlib.pyplot as plt
# import numpy as np
#
#
#
# todolist = [BMDataset3D1e9]
#
# for dataset_fun in todolist:
#     X, title = dataset_fun()
#     X = X.float()
#     print(X.shape)
#     my_values = {
#         'X':X.float(),
#     }
#     container = torch.jit.script(Container(my_values))
#     container.save("../faulty_data.pt")
