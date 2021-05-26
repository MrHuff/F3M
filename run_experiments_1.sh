#CUDA_VISIBLE_DEVICES=0 python experiments.py --idx=1
#wait
#CUDA_VISIBLE_DEVICES=0 python experiments.py --idx=3
#wait
#CUDA_VISIBLE_DEVICES=0 python experiments.py --idx=2
#CUDA_VISIBLE_DEVICES=4 python experiments.py --idx=4 &>> experiment_4.out
#CUDA_VISIBLE_DEVICES=4 python experiments.py --idx=5 &>> experiment_5.out
CUDA_VISIBLE_DEVICES=0 python experiments.py --idx=6 &>> experiment_6.out
CUDA_VISIBLE_DEVICES=1 python experiments.py --idx=7 &>> experiment_7.out
#CUDA_VISIBLE_DEVICES=0 python experiments.py --idx=8 &>> experiment_8.out
#CUDA_VISIBLE_DEVICES=7 python experiments.py --idx=9 &>> experiment_9.out


