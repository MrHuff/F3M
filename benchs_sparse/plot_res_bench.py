
import os, pickle
from sparse_datasets import PlotData
from FFMbench import PlotBench
import matplotlib.pyplot as plt

os.chdir("benchs_sparse/results")
for filename in os.listdir():
    if filename.endswith(".pkl"):
        f = open(filename, "rb")
        res = pickle.load(f)
        f.close()
        PlotData(res["X"])#, max_npoints=10000)
        plt.title(res["title"])
        plt.axis('equal')
        PlotBench(res)
plt.show()
