CUDA_VISIBLE_DEVICES=0 python experiments_2.py --idx=1
#CUDACXX=/usr/local/cuda-10.2/bin/nvcc CUDA_VISIBLE_DEVICES=1 python experiments_2.py --idx=2 &>> experiment_2_uniform_bench.out
CUDA_VISIBLE_DEVICES=1 python experiments_2.py --idx=3
#CUDA_VISIBLE_DEVICES=5 python experiments_2.py --idx=4 &>> experiment_2_normal_bench.out
CUDA_VISIBLE_DEVICES=1 python experiments_2.py --idx=5
#CUDACXX=/usr/local/cuda-10.2/bin/nvcc CUDA_VISIBLE_DEVICES=7 python experiments_2.py --idx=6 &>> experiment_2_data_bench.out


