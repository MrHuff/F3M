loop () {
  echo $1
  for file in high_dim_jobs/batch_$1/*
  do
    CUDA_VISIBLE_DEVICES="$1" python experiments_distributed.py --idx=8 --chunk=0 --fn="$file" --dn='larger_dims'
    #whatever you need with "$file"
  done
}
loop 0 &
loop 1 &
loop 2 &
loop 3 &
loop 4 &
loop 5 &
loop 6 &
loop 7 &




#for file in high_dim_jobs/batch_1/*
#do
#  CUDA_VISIBLE_DEVICES=1 python experiments_distributed.py --idx=8 --chunk=0 --fn="$file" --dn='larger_dims'
#  wait
#  #whatever you need with "$file"
#done
#
#for file in high_dim_jobs/batch_2/*
#do
#  CUDA_VISIBLE_DEVICES=2 python experiments_distributed.py --idx=8 --chunk=0 --fn="$file" --dn='larger_dims'
#  wait
#  #whatever you need with "$file"
#done
#
#for file in high_dim_jobs/batch_3/*
#do
#  CUDA_VISIBLE_DEVICES=3 python experiments_distributed.py --idx=8 --chunk=0 --fn="$file" --dn='larger_dims'
#  wait
#  #whatever you need with "$file"
#done
#
#for file in high_dim_jobs/batch_4/*
#do
#  CUDA_VISIBLE_DEVICES=4 python experiments_distributed.py --idx=8 --chunk=0 --fn="$file" --dn='larger_dims'
#  wait
#  #whatever you need with "$file"
#done
#
#for file in high_dim_jobs/batch_5/*
#do
#  CUDA_VISIBLE_DEVICES=5 python experiments_distributed.py --idx=8 --chunk=0 --fn="$file" --dn='larger_dims'
#  wait
#  #whatever you need with "$file"
#done
#
#for file in high_dim_jobs/batch_6/*
#do
#  CUDA_VISIBLE_DEVICES=6 python experiments_distributed.py --idx=8 --chunk=0 --fn="$file" --dn='larger_dims'
#  wait
#  #whatever you need with "$file"
#done
#
#for file in high_dim_jobs/batch_7/*
#do
#  CUDA_VISIBLE_DEVICES=7 python experiments_distributed.py --idx=8 --chunk=0 --fn="$file" --dn='larger_dims'
#  wait
#  #whatever you need with "$file"
#done

