n=1000000000
printf '%s\n' runtime n par_fac | paste -sd ',' >> "file_$n.csv"
start=`date +%s.%N`
CUDA_VISIBLE_DEVICES=0 python scalability_analysis.py --n=$n --d=3 --par_fac=1 --index=0 &
wait
end=`date +%s.%N`
runtime=$( echo "$end - $start" | bc -l )
printf '%s\n' $runtime $n 1 | paste -sd ',' >> "file_$n.csv"

start=`date +%s.%N`
CUDA_VISIBLE_DEVICES=0 python scalability_analysis.py --n=$n --d=3 --par_fac=2 --index=0 &
CUDA_VISIBLE_DEVICES=1 python scalability_analysis.py --n=$n --d=3 --par_fac=2 --index=1 &
wait
CUDA_VISIBLE_DEVICES=0 python scalability_analysis.py --n=$n --d=3 --par_fac=2 --index=2 &
CUDA_VISIBLE_DEVICES=1 python scalability_analysis.py --n=$n --d=3 --par_fac=2 --index=3 &
end=`date +%s.%N`
runtime=$( echo "$end - $start" | bc -l )
printf '%s\n' $runtime $n 2 | paste -sd ',' >> "file_$n.csv"

start=`date +%s.%N`
CUDA_VISIBLE_DEVICES=0 python scalability_analysis.py --n=$n --d=3 --par_fac=2 --index=0 &
CUDA_VISIBLE_DEVICES=1 python scalability_analysis.py --n=$n --d=3 --par_fac=2 --index=1 &
CUDA_VISIBLE_DEVICES=2 python scalability_analysis.py --n=$n --d=3 --par_fac=2 --index=2 &
CUDA_VISIBLE_DEVICES=3 python scalability_analysis.py --n=$n --d=3 --par_fac=2 --index=3 &
wait
end=`date +%s.%N`
runtime=$( echo "$end - $start" | bc -l )
printf '%s\n' $runtime $n 4 | paste -sd ',' >> "file_$n.csv"

start=`date +%s.%N`
CUDA_VISIBLE_DEVICES=0 python scalability_analysis.py --n=$n --d=3 --par_fac=4 --index=0 &
CUDA_VISIBLE_DEVICES=1 python scalability_analysis.py --n=$n --d=3 --par_fac=4 --index=1 &
CUDA_VISIBLE_DEVICES=2 python scalability_analysis.py --n=$n --d=3 --par_fac=4 --index=2 &
CUDA_VISIBLE_DEVICES=3 python scalability_analysis.py --n=$n --d=3 --par_fac=4 --index=3 &
CUDA_VISIBLE_DEVICES=4 python scalability_analysis.py --n=$n --d=3 --par_fac=4 --index=4 &
CUDA_VISIBLE_DEVICES=5 python scalability_analysis.py --n=$n --d=3 --par_fac=4 --index=5 &
CUDA_VISIBLE_DEVICES=6 python scalability_analysis.py --n=$n --d=3 --par_fac=4 --index=6 &
CUDA_VISIBLE_DEVICES=7 python scalability_analysis.py --n=$n --d=3 --par_fac=4 --index=7 &
wait
end=`date +%s.%N`
runtime=$( echo "$end - $start" | bc -l )
printf '%s\n' $runtime $n 8 | paste -sd ',' >> "file_$n.csv"
