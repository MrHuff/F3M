#CUDA_VISIBLE_DEVICES=0 python experiments_KMVM_real_datasets.py --dataset=taxi --ablation=0 --chunk=0 &
#CUDA_VISIBLE_DEVICES=1 python experiments_KMVM_real_datasets.py --dataset=taxi --ablation=0 --chunk=1 &
#CUDA_VISIBLE_DEVICES=2 python experiments_KMVM_real_datasets.py --dataset=taxi --ablation=0 --chunk=2 &
#CUDA_VISIBLE_DEVICES=3 python experiments_KMVM_real_datasets.py --dataset=taxi --ablation=0 --chunk=3 &
#CUDA_VISIBLE_DEVICES=4 python experiments_KMVM_real_datasets.py --dataset=taxi --ablation=0 --chunk=4 &
#CUDA_VISIBLE_DEVICES=5 python experiments_KMVM_real_datasets.py --dataset=taxi --ablation=0 --chunk=5 &
#CUDA_VISIBLE_DEVICES=6 python experiments_KMVM_real_datasets.py --dataset=taxi --ablation=0 --chunk=6 &
#CUDA_VISIBLE_DEVICES=7 python experiments_KMVM_real_datasets.py --dataset=taxi --ablation=0 --chunk=7 &
#wait
#CUDA_VISIBLE_DEVICES=0 python experiments_KMVM_real_datasets.py --dataset=osm --ablation=0 --chunk=0 &
#CUDA_VISIBLE_DEVICES=1 python experiments_KMVM_real_datasets.py --dataset=osm --ablation=0 --chunk=1 &
#CUDA_VISIBLE_DEVICES=2 python experiments_KMVM_real_datasets.py --dataset=osm --ablation=0 --chunk=2 &
#CUDA_VISIBLE_DEVICES=3 python experiments_KMVM_real_datasets.py --dataset=osm --ablation=0 --chunk=3 &
#CUDA_VISIBLE_DEVICES=4 python experiments_KMVM_real_datasets.py --dataset=osm --ablation=0 --chunk=4 &
#CUDA_VISIBLE_DEVICES=5 python experiments_KMVM_real_datasets.py --dataset=osm --ablation=0 --chunk=5 &
#CUDA_VISIBLE_DEVICES=6 python experiments_KMVM_real_datasets.py --dataset=osm --ablation=0 --chunk=6 &
#CUDA_VISIBLE_DEVICES=7 python experiments_KMVM_real_datasets.py --dataset=osm --ablation=0 --chunk=7 &
#wait
#CUDA_VISIBLE_DEVICES=0 python experiments_KMVM_real_datasets.py --dataset=osm --ablation=1 --chunk=0 &
#CUDA_VISIBLE_DEVICES=1 python experiments_KMVM_real_datasets.py --dataset=osm --ablation=1 --chunk=1 &
#CUDA_VISIBLE_DEVICES=2 python experiments_KMVM_real_datasets.py --dataset=osm --ablation=1 --chunk=2 &
#CUDA_VISIBLE_DEVICES=3 python experiments_KMVM_real_datasets.py --dataset=osm --ablation=1 --chunk=3 &
#CUDA_VISIBLE_DEVICES=4 python experiments_KMVM_real_datasets.py --dataset=osm --ablation=1 --chunk=4 &
#CUDA_VISIBLE_DEVICES=5 python experiments_KMVM_real_datasets.py --dataset=osm --ablation=1 --chunk=5 &
#CUDA_VISIBLE_DEVICES=6 python experiments_KMVM_real_datasets.py --dataset=osm --ablation=1 --chunk=6 &
#CUDA_VISIBLE_DEVICES=7 python experiments_KMVM_real_datasets.py --dataset=osm --ablation=1 --chunk=7 &
#wait
#CUDA_VISIBLE_DEVICES=0 python experiments_KMVM_real_datasets.py --dataset=taxi --ablation=1 --chunk=0 &
#CUDA_VISIBLE_DEVICES=1 python experiments_KMVM_real_datasets.py --dataset=taxi --ablation=1 --chunk=1 &
#CUDA_VISIBLE_DEVICES=2 python experiments_KMVM_real_datasets.py --dataset=taxi --ablation=1 --chunk=2 &
#CUDA_VISIBLE_DEVICES=3 python experiments_KMVM_real_datasets.py --dataset=taxi --ablation=1 --chunk=3 &
#CUDA_VISIBLE_DEVICES=4 python experiments_KMVM_real_datasets.py --dataset=taxi --ablation=1 --chunk=4 &
#CUDA_VISIBLE_DEVICES=5 python experiments_KMVM_real_datasets.py --dataset=taxi --ablation=1 --chunk=5 &
#CUDA_VISIBLE_DEVICES=6 python experiments_KMVM_real_datasets.py --dataset=taxi --ablation=1 --chunk=6 &
#CUDA_VISIBLE_DEVICES=7 python experiments_KMVM_real_datasets.py --dataset=taxi --ablation=1 --chunk=7 &
#wait
#CUDACXX=/opt/cuda-10.2/bin/nvcc CUDA_VISIBLE_DEVICES=0 python experiments_krr_b.py --idx=3 --penalty=1e-5 --seed=1 &
#CUDACXX=/opt/cuda-10.2/bin/nvcc CUDA_VISIBLE_DEVICES=1 python experiments_krr_b.py --idx=3 --penalty=1e-2 --seed=2 &
#CUDACXX=/opt/cuda-10.2/bin/nvcc CUDA_VISIBLE_DEVICES=2 python experiments_krr_b.py --idx=3 --penalty=1e-2 --seed=3 &
#CUDACXX=/opt/cuda-10.2/bin/nvcc CUDA_VISIBLE_DEVICES=3 python experiments_krr_b.py --idx=2 --penalty=1e-5 --seed=1 &
#CUDACXX=/opt/cuda-10.2/bin/nvcc CUDA_VISIBLE_DEVICES=5 python experiments_krr_b.py --idx=2 --penalty=1e-5 --seed=2 &
#CUDACXX=/opt/cuda-10.2/bin/nvcc CUDA_VISIBLE_DEVICES=4 python experiments_krr_b.py --idx=3 --penalty=1e-2 --seed=1 &
#CUDACXX=/opt/cuda-10.2/bin/nvcc CUDA_VISIBLE_DEVICES=5 python experiments_krr_b.py --idx=3 --penalty=1e-2 --seed=2 &
#CUDA_VISIBLE_DEVICES=2 python experiments_78D.py --idx=1 &
#CUDA_VISIBLE_DEVICES=3 python experiments_78D.py --idx=2 &
#CUDA_VISIBLE_DEVICES=4 python experiments_78D.py --idx=3 &




##CUDA_VISIBLE_DEVICES=2 python ablation_experiments.py --idx=1 &
##CUDA_VISIBLE_DEVICES=3 python ablation_experiments.py --idx=3 &
##CUDA_VISIBLE_DEVICES=4 python ablation_experiments.py --idx=2 &
#CUDA_VISIBLE_DEVICES=0 python experiments_distributed.py --idx=6 --chunk=0 &
#CUDA_VISIBLE_DEVICES=1 python experiments_distributed.py --idx=6 --chunk=1 &
#CUDA_VISIBLE_DEVICES=2 python experiments_distributed.py --idx=6 --chunk=2 &
#CUDA_VISIBLE_DEVICES=3 python experiments_distributed.py --idx=6 --chunk=3 &
#CUDA_VISIBLE_DEVICES=4 python experiments_distributed.py --idx=6 --chunk=4 &
#CUDA_VISIBLE_DEVICES=5 python experiments_distributed.py --idx=6 --chunk=5 &
#CUDA_VISIBLE_DEVICES=6 python experiments_distributed.py --idx=6 --chunk=6 &
#CUDA_VISIBLE_DEVICES=7 python experiments_distributed.py --idx=6 --chunk=7 &
#wait
#CUDA_VISIBLE_DEVICES=0 python experiments_distributed.py --idx=7 --chunk=0 &
#CUDA_VISIBLE_DEVICES=1 python experiments_distributed.py --idx=7 --chunk=1 &
#CUDA_VISIBLE_DEVICES=2 python experiments_distributed.py --idx=7 --chunk=2 &
#CUDA_VISIBLE_DEVICES=3 python experiments_distributed.py --idx=7 --chunk=3 &
#CUDA_VISIBLE_DEVICES=4 python experiments_distributed.py --idx=7 --chunk=4 &
#CUDA_VISIBLE_DEVICES=5 python experiments_distributed.py --idx=7 --chunk=5 &
#CUDA_VISIBLE_DEVICES=6 python experiments_distributed.py --idx=7 --chunk=6 &
#CUDA_VISIBLE_DEVICES=7 python experiments_distributed.py --idx=7 --chunk=7 &
#wait
CUDA_VISIBLE_DEVICES=0 python experiments_distributed.py --idx=8 --chunk=0 --fn='uniform_jobs' --dn='larger_dims_uniform'&
CUDA_VISIBLE_DEVICES=1 python experiments_distributed.py --idx=8 --chunk=1 --fn='uniform_jobs' --dn='larger_dims_uniform'&
CUDA_VISIBLE_DEVICES=2 python experiments_distributed.py --idx=8 --chunk=2 --fn='uniform_jobs' --dn='larger_dims_uniform'&
CUDA_VISIBLE_DEVICES=3 python experiments_distributed.py --idx=8 --chunk=3 --fn='uniform_jobs' --dn='larger_dims_uniform'&
CUDA_VISIBLE_DEVICES=4 python experiments_distributed.py --idx=8 --chunk=4 --fn='uniform_jobs' --dn='larger_dims_uniform'&
CUDA_VISIBLE_DEVICES=5 python experiments_distributed.py --idx=8 --chunk=5 --fn='uniform_jobs' --dn='larger_dims_uniform'&
CUDA_VISIBLE_DEVICES=6 python experiments_distributed.py --idx=8 --chunk=6 --fn='uniform_jobs' --dn='larger_dims_uniform'&
CUDA_VISIBLE_DEVICES=7 python experiments_distributed.py --idx=8 --chunk=7 --fn='uniform_jobs' --dn='larger_dims_uniform'&

#
wait
CUDA_VISIBLE_DEVICES=0 python experiments_distributed.py --idx=8 --chunk=0 --fn='normal_jobs' --dn='larger_dims_normal'&
CUDA_VISIBLE_DEVICES=1 python experiments_distributed.py --idx=8 --chunk=1 --fn='normal_jobs' --dn='larger_dims_normal'&
CUDA_VISIBLE_DEVICES=2 python experiments_distributed.py --idx=8 --chunk=2 --fn='normal_jobs' --dn='larger_dims_normal'&
CUDA_VISIBLE_DEVICES=3 python experiments_distributed.py --idx=8 --chunk=3 --fn='normal_jobs' --dn='larger_dims_normal'&
CUDA_VISIBLE_DEVICES=4 python experiments_distributed.py --idx=8 --chunk=4 --fn='normal_jobs' --dn='larger_dims_normal'&
CUDA_VISIBLE_DEVICES=5 python experiments_distributed.py --idx=8 --chunk=5 --fn='normal_jobs' --dn='larger_dims_normal'&
CUDA_VISIBLE_DEVICES=6 python experiments_distributed.py --idx=8 --chunk=6 --fn='normal_jobs' --dn='larger_dims_normal'&
CUDA_VISIBLE_DEVICES=7 python experiments_distributed.py --idx=8 --chunk=7 --fn='normal_jobs' --dn='larger_dims_normal'&
wait
CUDA_VISIBLE_DEVICES=0 python experiments_distributed.py --idx=8 --chunk=0 --fn='mix_jobs' --dn='larger_dims_mix'&
CUDA_VISIBLE_DEVICES=1 python experiments_distributed.py --idx=8 --chunk=1 --fn='mix_jobs' --dn='larger_dims_mix'&
CUDA_VISIBLE_DEVICES=2 python experiments_distributed.py --idx=8 --chunk=2 --fn='mix_jobs' --dn='larger_dims_mix'&
CUDA_VISIBLE_DEVICES=3 python experiments_distributed.py --idx=8 --chunk=3 --fn='mix_jobs' --dn='larger_dims_mix'&
CUDA_VISIBLE_DEVICES=4 python experiments_distributed.py --idx=8 --chunk=4 --fn='mix_jobs' --dn='larger_dims_mix'&
CUDA_VISIBLE_DEVICES=5 python experiments_distributed.py --idx=8 --chunk=5 --fn='mix_jobs' --dn='larger_dims_mix'&
CUDA_VISIBLE_DEVICES=6 python experiments_distributed.py --idx=8 --chunk=6 --fn='mix_jobs' --dn='larger_dims_mix'&
CUDA_VISIBLE_DEVICES=7 python experiments_distributed.py --idx=8 --chunk=7 --fn='mix_jobs' --dn='larger_dims_mix'&