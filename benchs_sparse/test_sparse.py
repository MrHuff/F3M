import matplotlib
matplotlib.use("MacOSX")
import matplotlib.pyplot as plt
#interactive backends: GTK3Agg, GTK3Cairo, MacOSX, nbAgg, Qt4Agg, Qt4Cairo, Qt5Agg, Qt5Cairo, TkAgg, TkCairo, WebAgg, WX, WXAgg, WXCairo
#non-interactive backends: agg, cairo, pdf, pgf, ps, svg, template

from sparse_datasets import PlotData, MaternClusterData, FBMData

X = MaternClusterData(2,100,100,.05)
PlotData(X)

X = MaternClusterData(3,100,100,.05)
PlotData(X)

n = 10000
X = FBMData(2,n,0.25)
PlotData(X)

X = FBMData(3,n,0.25)
PlotData(X)

plt.show()

