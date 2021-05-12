
import os, pickle
from sparse_datasets import PlotData
from FFMbench import PlotBench
import matplotlib.pyplot as plt

for filename in os.listdir("results"):
    if filename.endswith(".pkl"):
        f = open("results/"+filename, "rb")
        res = pickle.load(f)
        f.close()
        PlotData(res["X"], max_npoints=1000000)
        plt.title(res["title"])
        PlotBench(res)
        print()
        print(res["title"])
        print("best time : ", min(res["elapsed"]))
        print("best accuracy : ", min(res["rel_err"]))
        print("total time for FFM computations : ", sum(res["elapsed"]))
plt.show()
