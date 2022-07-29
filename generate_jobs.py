import pickle
import os
D_LIST = [3]
def generate_jobs_normal():
    counter=0
    dict_list = []
    for seed in [0]:
        for d in D_LIST:
            for nr_of_interpolation in [3,4,5]:
                nr_of_interpolation = nr_of_interpolation**d
                for r2 in [0.1, 1, 10]:
                    for eff_var_limit in [0.1,0.3,0.5]:
                        for n, min_points, small_field_limit in zip([1000000, 10000000, 100000000,1000000000],
                                                                    [1000, 1000, 1000,5000],
                                                                    [nr_of_interpolation, nr_of_interpolation,
                                                                     nr_of_interpolation,nr_of_interpolation]):
                            if d > 4 and n == 500000000:
                                n = 500000000 // 2
                            dict_param = {
                                'eff_var':seed,
                                'd':int(d),
                                'nr_of_interpolation':nr_of_interpolation,
                                'n':int(n),
                                'min_points':min_points,
                                'small_field_limit':small_field_limit,
                                'r2':r2,
                                'eff_var_limit':d*eff_var_limit,
                                'counter':counter,
                                'mode':2
                            }
                            dict_list.append(dict_param)
                            counter+=1
    with open('normal_jobs.pkl', 'wb') as f:
        pickle.dump(dict_list, f)
    return dict_list

def generate_jobs_uniform():
    counter=0
    dict_list = []
    for seed in [0]:
        for d in D_LIST:
            for nr_of_interpolation in [3,4,5]:
                nr_of_interpolation = nr_of_interpolation**d
                for r2 in [0.1, 1, 10]:
                    for eff_var_limit in [0.1,0.3,0.5]:
                        for n, min_points, small_field_limit in zip([1000000, 10000000, 100000000,1000000000],
                                                                    [1000, 1000, 1000,5000],
                                                                    [nr_of_interpolation, nr_of_interpolation,
                                                                     nr_of_interpolation,nr_of_interpolation]):
                            dict_param = {
                                'eff_var':seed,
                                'd':d,
                                'nr_of_interpolation':nr_of_interpolation,
                                'n':n,
                                'min_points':min_points,
                                'small_field_limit':small_field_limit,
                                'r2':r2,
                                'eff_var_limit':d*eff_var_limit,
                                'counter': counter,
                                'mode': 1

                            }
                            dict_list.append(dict_param)
                            counter+=1
    with open('uniform_jobs.pkl', 'wb') as f:
        pickle.dump(dict_list, f)
    return dict_list

def generate_jobs_mix():
    counter=0
    dict_list = []
    for seed in [0]:
        for d in D_LIST:
            for nr_of_interpolation in [3,4,5]:
                nr_of_interpolation = nr_of_interpolation**d
                for r2 in [0.1, 1, 10]:
                    for eff_var_limit in [0.1,0.3,0.5]:
                        for n, min_points, small_field_limit in zip([1000000, 10000000, 100000000,1000000000],
                                                                    [1000, 1000, 1000,5000],
                                                                    [nr_of_interpolation, nr_of_interpolation,
                                                                     nr_of_interpolation,nr_of_interpolation]):
                            dict_param = {
                                'eff_var':seed,
                                'd':d,
                                'nr_of_interpolation':nr_of_interpolation,
                                'n':n,
                                'min_points':min_points,
                                'small_field_limit':small_field_limit,
                                'r2':r2,
                                'eff_var_limit':d*eff_var_limit,
                                'counter': counter,
                                'mode': 3

                            }
                            dict_list.append(dict_param)
                            counter+=1
    with open('mix_jobs.pkl', 'wb') as f:
        pickle.dump(dict_list, f)
    return dict_list

def generate_jobs_real_KMVM():
    node_list = [4,5,6]
    counter=0
    dict_list = []
    dict_list_ablation=[]
    counter_ablation=0
    for s in [1]:
            for r2 in [0.1, 1, 10]:
                for node_nr in node_list:

                    for n, min_points in zip([1000000, 10000000, 100000000, 1000000000],
                                             [1000, 1000, 5000, 5000]):
                        dict_param_ablation = {
                            'node_nr': node_nr,
                            'seed': s,
                            'n': n,
                            'min_points': min(n//20,500000),
                            'r2': r2,
                            'eff_var_limit': 1.0,
                            'counter': counter_ablation
                        }
                        dict_list_ablation.append(dict_param_ablation)
                        for evarlimit in [0.1, 0.3, 0.5]:
                            dict_param = {
                                'node_nr':node_nr,
                                'seed': s,
                                'n': n,
                                'min_points': min_points,
                                'r2': r2,
                                'eff_var_limit': evarlimit,
                                'counter': counter
                            }
                            dict_list.append(dict_param)
                            counter += 1
                        counter_ablation+=1
    with open('real_kmvm_jobs.pkl', 'wb') as f:
        pickle.dump(dict_list, f)
    with open('real_kmvm_jobs_ablation.pkl', 'wb') as f:
        pickle.dump(dict_list_ablation, f)
if __name__ == '__main__':
    # generate_jobs_real_KMVM()
    a=generate_jobs_uniform()
    b=generate_jobs_normal()
    c= generate_jobs_mix()
    all_jobs = a+b+c
    NR_OF_GPUS=8
    os_name = '3d_jobs_25'
    for i in range(NR_OF_GPUS):
        fn = f'{os_name}/batch_{i}'
        if not os.path.exists(fn):
            os.makedirs(fn)

    sorted_jobs = sorted(all_jobs, key=lambda x: x['n'], reverse=False)
    print(len(sorted_jobs))
    for i,el in enumerate(sorted_jobs):
        el['counter']=i
        print(el['counter'])
        box = i%8
        fn_i = f'{os_name}/batch_{box}/job_{i}.pkl'
        with open(fn_i, 'wb') as f:
            pickle.dump(el, f)
