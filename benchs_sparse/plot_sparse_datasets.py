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
# X = MaternClusterData(2,100,100,.05)
# PlotData(X)
#
# X = MaternClusterData(3,100,100,.05)
# PlotData(X)

n = 10000
X = MultiScaleClusterData(3,[10,10,10],.03)
PlotData(X)

# X = FBMData(3,n,0.25)
# PlotData(X)

plt.show()

