*3F-M* (Triple-FM)

To able to run anything with 3F-M you will need:

1. A pytorch v1.8>= with either CUDA 10.2 or 11.1.
2. A GCC compiler. It's recommended to use the https://anaconda.org/omgarcia/gcc-6 compiler that comes with conda, it works for all dependencies.
3. CUDA 10.2 nvcc compiler, for pykeops and FALKON.

To run 3,4,5D experiments run the command:

python experiments.py --idx={experiment number}

To run FALKON experiments: 

1. First generate the data by running python generate_KRR_data.py
2. Then run python experiments_2.py --idx={experiment number}


