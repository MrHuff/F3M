CUDA_VISIBLE_DEVICES=2 python ablation_experiments.py --idx=1 &
CUDA_VISIBLE_DEVICES=3 python ablation_experiments.py --idx=3 &
CUDA_VISIBLE_DEVICES=4 python ablation_experiments.py --idx=2 &
CUDA_VISIBLE_DEVICES=5 python experiment678.py --idx=6 &
CUDA_VISIBLE_DEVICES=6 python experiment678.py --idx=7 &
CUDA_VISIBLE_DEVICES=7 python experiment678.py --idx=8 &
#CUDA_VISIBLE_DEVICES=4 python experiments.py --idx=4 &>> experiment_4.out
#CUDA_VISIBLE_DEVICES=4 python experiments.py --idx=5 &>> experiment_5.out

#CUDA_VISIBLE_DEVICES=0 python experiments.py --idx=8 &>> experiment_8_hack.out
#CUDA_VISIBLE_DEVICES=7 python experiments.py --idx=9 &>> experiment_9.out
#CUDA_VISIBLE_DEVICES=0 python hail_mary.py --idx=0 &>> experiment_1.out
#CUDA_VISIBLE_DEVICES=1 python hail_mary.py --idx=1 &>> experiment_2.out
#CUDA_VISIBLE_DEVICES=2 python hail_mary.py --idx=2 &>> experiment_3.out
#CUDA_VISIBLE_DEVICES=3 python hail_mary.py --idx=3 &>> experiment_4.out
#CUDA_VISIBLE_DEVICES=4 python hail_mary.py --idx=4 &>> experiment_5.out
#CUDA_VISIBLE_DEVICES=5 python hail_mary.py --idx=5 &>> experiment_6.out
#CUDA_VISIBLE_DEVICES=6 python hail_mary.py --idx=6 &>> experiment_7.out
#CUDA_VISIBLE_DEVICES=7 python hail_mary.py --idx=7 &>> experiment_8.out

