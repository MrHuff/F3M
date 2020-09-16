for i in {3..7}
do
  for j in {100000,1000000}
  do
      for k in {250,500,1000}
      do
        ./cmake-build-release/cuda_n_body $i $j $k
      done
  done
done
