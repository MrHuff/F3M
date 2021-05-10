
# install fbm via
pip install git+https://github.com/joanglaunes/fbm.git@avoid_loops

# run benchmarks :
python bench_sparse_datasets.py 

# plot benchmark results :
python plot_res_bench.py




# best params so far : 
# for ClusteredDataset2D : 64, 0.35, 500
# for FBMDataset2D : 64, 0.25, 1000
# for BMDataset2D : 64, 0.3, 1000 or 128, 0.35, 2000
