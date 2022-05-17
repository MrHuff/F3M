#CUDA_VISIBLE_DEVICES=5 python experiments_krr_a.py --idx=1 &>> experiment_2_uniform.out
#CUDACXX=/usr/local/cuda-10.2/bin/nvcc CUDA_VISIBLE_DEVICES=2,3 python experiments_krr_a.py --idx=2 &>> experiment_2_uniform_bench.out
#CUDA_VISIBLE_DEVICES=4 python experiments_krr_a.py --idx=3 &>> experiment_2_normal.out
#CUDACXX=/usr/local/cuda-10.2/bin/nvcc CUDA_VISIBLE_DEVICES=5 python experiments_krr_a.py --idx=4 &>> experiment_2_normal_bench.out
#CUDA_VISIBLE_DEVICES=5 python experiments_krr_a.py --idx=5 &>> experiment_2_dataset.out
#CUDACXX=/usr/local/cuda-10.2/bin/nvcc CUDA_VISIBLE_DEVICES=6,7 python experiments_krr_a.py --idx=6 &>> experiment_2_data_bench.out
#CUDA_VISIBLE_DEVICES=2 python experiments_krr_a.py --idx=1 &>> experiment_2_uniform.out
#CUDA_VISIBLE_DEVICES=6 python experiments_krr_a.py --idx=5 &>> experiment_2_dataset.out
#CUDACXX=/usr/local/cuda-10.2/bin/nvcc CUDA_VISIBLE_DEVICES=7 python experiments_krr_a.py --idx=6 &>> experiment_2_data_bench.out


#CUDACXX=/opt/cuda-10.2/bin/nvcc CUDA_VISIBLE_DEVICES=0 python experiments_krr_b.py --idx=0 --penalty_in=1e-1 --eff_var=0.1 &
#CUDACXX=/opt/cuda-10.2/bin/nvcc CUDA_VISIBLE_DEVICES=0 python experiments_krr_b.py --idx=0 --penalty_in=7.5e-3 --eff_var=1 &
CUDA_VISIBLE_DEVICES=0 python run_gp_experiments_multi_gpu.py --chunk_idx=0 --job_folder=job_f3m &
CUDA_VISIBLE_DEVICES=1 python run_gp_experiments_multi_gpu.py --chunk_idx=1 --job_folder=job_f3m &
CUDA_VISIBLE_DEVICES=2 python run_gp_experiments_multi_gpu.py --chunk_idx=2 --job_folder=job_f3m &
CUDA_VISIBLE_DEVICES=3 python run_gp_experiments_multi_gpu.py --chunk_idx=3 --job_folder=job_f3m &
CUDA_VISIBLE_DEVICES=4 python run_gp_experiments_multi_gpu.py --chunk_idx=4 --job_folder=job_f3m &
CUDA_VISIBLE_DEVICES=5 python run_gp_experiments_multi_gpu.py --chunk_idx=5 --job_folder=job_f3m &
CUDA_VISIBLE_DEVICES=6 python run_gp_experiments_multi_gpu.py --chunk_idx=6 --job_folder=job_f3m &
CUDA_VISIBLE_DEVICES=7 python run_gp_experiments_multi_gpu.py --chunk_idx=7 --job_folder=job_f3m &