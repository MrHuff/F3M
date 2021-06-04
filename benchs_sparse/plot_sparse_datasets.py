import matplotlib
import platform
if platform.system()=='Linux':
    matplotlib.use("Qt5Agg")
elif platform.system()=='Darwin':
    matplotlib.use("MacOSX")
#interactive backends: GTK3Agg, GTK3Cairo, MacOSX, nbAgg, Qt4Agg, Qt4Cairo, Qt5Agg, Qt5Cairo, TkAgg, TkCairo, WebAgg, WX, WXAgg, WXCairo
#non-interactive backends: agg, cairo, pdf, pgf, ps, svg, template

import matplotlib.pyplot as plt

from sparse_datasets import *
import random
import numpy as np
import torch
random.seed(0)
np.random.seed(0)
torch.manual_seed(1)
n=5000
X = MaternClusterData(2,100,100,.05)
PlotDataSave(X,name='cluster.png')
X = FBMData(2,n,.75)
PlotDataSave(X,name='FBM.png')
X = FBMData(2,n,.5)
PlotDataSave(X,name='BM.png')
X = torch.rand(5000,2)*12**0.5
PlotDataSave(X,name='unif.png')
X = torch.randn(5000,2)
PlotDataSave(X,name='normal.png')

X = torch.rand(5000,2)*12**0.5
Y = torch.randn(5000,2)
PlotDataSave2(X,Y,name='mix.png')
#
# X = MaternClusterData(3,100,100,.05)
# PlotData(X)

# n = 100000000
# X = FBMData(3,n,.75)
# print(X.shape)
# PlotData(X)

# X = FBMData(3,n,0.25)
# PlotData(X)

# plt.show()

