
import os, pickle
from sparse_datasets import PlotData
from FFMbench import PlotBench
import matplotlib.pyplot as plt

for filename in os.listdir("results"):
    if filename.endswith(".pkl"):
        f = open("results/"+filename, "rb")
        res = pickle.load(f)
        f.close()
        PlotData(res["X"])#, max_npoints=10000)
        plt.title(res["title"])
        plt.axis('equal')
        PlotBench(res)
plt.show()
