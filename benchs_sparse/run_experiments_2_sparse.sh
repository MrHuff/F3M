#CUDACXX=/opt/cuda-10.2/bin/nvcc CUDA_VISIBLE_DEVICES=0 python bench_sparse_datasets.py BMDataset3D
#CUDACXX=/opt/cuda-10.2/bin/nvcc CUDA_VISIBLE_DEVICES=0 python bench_sparse_datasets.py FBMDataset3D
#CUDACXX=/opt/cuda-10.2/bin/nvcc CUDA_VISIBLE_DEVICES=0 python bench_sparse_datasets.py ClusteredDataset3D

CUDACXX=/opt/cuda-10.2/bin/nvcc CUDA_VISIBLE_DEVICES=0 python bench_sparse_datasets_25.py BMDataset3D
CUDACXX=/opt/cuda-10.2/bin/nvcc CUDA_VISIBLE_DEVICES=0 python bench_sparse_datasets_25.py FBMDataset3D
CUDACXX=/opt/cuda-10.2/bin/nvcc CUDA_VISIBLE_DEVICES=0 python bench_sparse_datasets_25.py ClusteredDataset3D
