from benchs_sparse.sparse_datasets import *
import random
import numpy as np
from FFM_classes import *

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
class Container(torch.nn.Module):
    def __init__(self, my_values):
        super().__init__()
        for key in my_values:
            setattr(self, key, my_values[key])


dat, title = ClusteredDataset3D1e7()
my_values = {
    'X':dat.float()
}

# Save arbitrary values supported by TorchScript
# https://pytorch.org/docs/master/jit.html#supported-type
container = torch.jit.script(Container(my_values))
container.save("clustered_data.pt")

ffm_obj = FFM(X=dat,eff_var_limit=0.1,var_compression=True,min_points=5000)
b = torch.randn(dat.shape[0],1)

res = ffm_obj@b
