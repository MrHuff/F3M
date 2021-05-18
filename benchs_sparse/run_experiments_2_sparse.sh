CUDACXX=/usr/local/cuda-10.2/bin/nvcc CUDA_VISIBLE_DEVICES=0 python bench_sparse_datasets.py FBMDataset3D &>> experiment_1_sparse.out
CUDACXX=/usr/local/cuda-10.2/bin/nvcc CUDA_VISIBLE_DEVICES=1 python bench_sparse_datasets.py BMDataset3D &>> experiment_2_sparse.out
CUDACXX=/usr/local/cuda-10.2/bin/nvcc CUDA_VISIBLE_DEVICES=6 python bench_sparse_datasets.py ClusteredDataset3D &>> experiment_3_sparse.out


