from run_obj import *


if __name__ == '__main__':

    args_in={
        'job_index':0,
        'save_path':'test',
        'N':1000000,
        'd':3,
        'min_points':1000,
        'ref_points':5000,
        'ls':1.0,
        'nr_of_interpolation':64,
        'dist_1':'uniform',
        'dist_1_a':0.0,
        'dist_1_b':1.0,
        'var_limit':0.2,
        'var_comp':False ,
        'smooth':True,
        'seed':1337,
        'n_loops':10,
        'dist_2':None,
        'dist_2_a':None,
        'dist_2_b':None
    }
    run_and_record_simulated(args_in=args_in)



