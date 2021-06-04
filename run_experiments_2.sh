#CUDA_VISIBLE_DEVICES=5 python experiments_2.py --idx=1 &>> experiment_2_uniform.out
#CUDACXX=/usr/local/cuda-10.2/bin/nvcc CUDA_VISIBLE_DEVICES=2,3 python experiments_2.py --idx=2 &>> experiment_2_uniform_bench.out
#CUDA_VISIBLE_DEVICES=4 python experiments_2.py --idx=3 &>> experiment_2_normal.out
#CUDACXX=/usr/local/cuda-10.2/bin/nvcc CUDA_VISIBLE_DEVICES=5 python experiments_2.py --idx=4 &>> experiment_2_normal_bench.out
#CUDA_VISIBLE_DEVICES=5 python experiments_2.py --idx=5 &>> experiment_2_dataset.out
#CUDACXX=/usr/local/cuda-10.2/bin/nvcc CUDA_VISIBLE_DEVICES=6,7 python experiments_2.py --idx=6 &>> experiment_2_data_bench.out
#CUDA_VISIBLE_DEVICES=2 python experiments_2.py --idx=1 &>> experiment_2_uniform.out
#CUDA_VISIBLE_DEVICES=6 python experiments_2.py --idx=5 &>> experiment_2_dataset.out
#CUDACXX=/usr/local/cuda-10.2/bin/nvcc CUDA_VISIBLE_DEVICES=7 python experiments_2.py --idx=6 &>> experiment_2_data_bench.out


#CUDACXX=/opt/cuda-10.2/bin/nvcc CUDA_VISIBLE_DEVICES=0 python experiments_3.py --idx=0 --penalty_in=1e-1 --eff_var=0.1 &
#CUDACXX=/opt/cuda-10.2/bin/nvcc CUDA_VISIBLE_DEVICES=0 python experiments_3.py --idx=0 --penalty_in=7.5e-3 --eff_var=1 &
CUDACXX=/opt/cuda-10.2/bin/nvcc CUDA_VISIBLE_DEVICES=1 python experiments_3.py --idx=1 --penalty_in=5e-4 --seed=1 &
CUDACXX=/opt/cuda-10.2/bin/nvcc CUDA_VISIBLE_DEVICES=2 python experiments_3.py --idx=1 --penalty_in=5e-4 --seed=2 &
CUDACXX=/opt/cuda-10.2/bin/nvcc CUDA_VISIBLE_DEVICES=3 python experiments_3.py --idx=1 --penalty_in=5e-4 --seed=3 &
CUDACXX=/opt/cuda-10.2/bin/nvcc CUDA_VISIBLE_DEVICES=4 python experiments_3.py --idx=1 --penalty_in=2.5e-2 --seed=2 &
