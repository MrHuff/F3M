import os
import pickle
import shutil
def save_obj(obj, name ,folder):
    with open(f'{folder}'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name,folder):
    with open(f'{folder}' + name, 'rb') as f:
        return pickle.load(f)

def generate_XX_job(job_name,n_loops,ref_points,dist,dist_a,dist_b,ls_list,smooth,var_comp,var_limits=[]):
    if not os.path.exists(job_name):
        os.makedirs(job_name)
    else:
        shutil.rmtree(job_name)
        os.makedirs(job_name)
    args_in={
        'job_index':0,
        'save_path':f'{job_name}_results',
        'N':10000000,
        'd':2,
        'min_points':1000,
        'ref_points':ref_points,
        'ls':1.0,
        'nr_of_interpolation':16,
        'dist_1':'uniform',
        'dist_1_a':0.0,
        'dist_1_b':1.0,
        'var_limit':0.2,
        'var_comp':var_comp,
        'smooth':smooth,
        'seed':1337,
        'n_loops':n_loops,
        'dist_2':None,
        'dist_2_a':None,
        'dist_2_b':None
    }
    min_points_list = [1000,2500,5000]
    seed_list = [1337]
    N_list = [1e6,1e7,1e8]
    d_list = [1,2,3]
    counter = 0
    for min_p in min_points_list:
        for ls in ls_list:
            for seed in seed_list:
                for N in N_list:
                    for d in d_list:
                        for v in var_limits:
                            args = args_in
                            args['N']=N
                            args['job_index']=counter
                            args['d']=d
                            args['ls']=ls
                            args['nr_of_interpolation']=4**d
                            args['min_points']=min_p
                            args['var_limit']=v
                            args['seed']=seed
                            args['dist_1']=dist
                            args['dist_1_a']=dist_a
                            args['dist_1_b']=dist_b
                            save_obj(obj=args,name=f'/job_{counter}',folder = job_name)
                            counter+=1

def generate_XY_job(job_name,n_loops,ref_points,dist_1,dist_a_1,dist_b_1,dist_2,dist_a_2,dist_b_2):
    if not os.path.exists(job_name):
        os.makedirs(job_name)
    else:
        shutil.rmtree(job_name)
        os.makedirs(job_name)
    args_in={
        'job_index':0,
        'save_path':f'{job_name}_results',
        'N':10000000,
        'd':2,
        'min_points':1000,
        'ref_points':ref_points,
        'ls':1.0,
        'nr_of_interpolation':16,
        'dist_1':'uniform',
        'dist_1_a':0.0,
        'dist_1_b':1.0,
        'var_limit':0.2,
        'var_comp':False,
        'smooth':False,
        'seed':1337,
        'n_loops':n_loops,
        'dist_2':None,
        'dist_2_a':None,
        'dist_2_b':None
    }
    min_points_list = [1000,2500,5000]
    ls_list = [1e-6,1e-3,1e-1,1.0,10.,1e3,1e6]

    seed_list = [1337]
    N_list = [1e6,1e7,1e8]
    d_list = [1,2,3]
    counter = 0
    for min_p in min_points_list:
        for ls in ls_list:
            for seed in seed_list:
                for N in N_list:
                    for d in d_list:
                        args = args_in
                        args['N']=N
                        args['job_index']=counter
                        args['d']=d
                        args['ls']=ls
                        args['nr_of_interpolation']=4**d
                        args['min_points']=min_p
                        args['var_limit']=None
                        args['seed']=seed
                        args['dist_1']=dist_1
                        args['dist_1_a']=dist_a_1
                        args['dist_1_b']=dist_b_1
                        args['dist_2']=dist_2
                        args['dist_2_a']=dist_a_2
                        args['dist_2_b']=dist_b_2
                        save_obj(obj=args,name=f'/job_{counter}',folder = job_name)
                        counter+=1

if __name__ == '__main__':
    ls_list = [1e-3,1e-1,1.0,10,1e3]
    generate_XX_job(
        job_name='uniform_X_0_1',
                    n_loops=10,
                    ref_points=5000,
                    dist='uniform',
                    dist_a=0,
                    dist_b=1,
                    ls_list=ls_list,
                    smooth=False,
                    var_comp=False,
                    var_limits=[0.0]
                    )
    generate_XY_job(job_name='test_job_XY',n_loops=10,ref_points=5000,
                    dist_1='uniform',dist_a_1=0,dist_b_1=1
                    ,dist_2='normal',dist_a_2=0,dist_b_2=1)

