import pickle
import pandas as pd

def generate_jobs_normal():
    counter=0
    dict_list = []
    for seed in [0]:
        for d in [4, 5]:
            for nr_of_interpolation in [256,512,1024]:
                for r2 in [0.1, 1, 10]:
                    for eff_var_limit in [0.1,0.3,0.5]:
                        for n, min_points, small_field_limit in zip([1000000, 10000000, 100000000, 500000000],
                                                                    [5000, 5000, 20000, 20000],
                                                                    [nr_of_interpolation, 2500,
                                                                     10000, 15000]):
                            if d > 4 and n == 500000000:
                                n = 500000000 // 2
                            dict_param = {
                                'seed':seed,
                                'd':int(d),
                                'nr_of_interpolation':nr_of_interpolation,
                                'n':int(n),
                                'min_points':min_points,
                                'small_field_limit':small_field_limit,
                                'r2':r2,
                                'eff_var_limit':eff_var_limit,
                                'counter':counter
                            }
                            dict_list.append(dict_param)
                            counter+=1
    with open('normal_jobs.pkl', 'wb') as f:
        pickle.dump(dict_list, f)
def generate_jobs_uniform():
    counter=0
    dict_list = []
    for seed in [0]:
        for d in [4, 5]:
            for r_node in [4,5]:
                nr_of_interpolation = r_node**d
                for r2 in [0.1, 1, 10]:
                    for eff_var_limit in [0.5,1.0,2.0]:
                        for n, min_points, small_field_limit in zip([1000000, 10000000, 100000000, 500000000],
                                                                    [2500, 2500, 10000, 20000],
                                                                    [nr_of_interpolation, nr_of_interpolation,
                                                                     nr_of_interpolation, nr_of_interpolation]):
                            dict_param = {
                                'seed':seed,
                                'd':d,
                                'nr_of_interpolation':nr_of_interpolation,
                                'n':n,
                                'min_points':min_points,
                                'small_field_limit':small_field_limit,
                                'r2':r2,
                                'eff_var_limit':eff_var_limit,
                                'counter': counter

                            }
                            dict_list.append(dict_param)
                            counter+=1
    with open('uniform_jobs.pkl', 'wb') as f:
        pickle.dump(dict_list, f)

if __name__ == '__main__':
    generate_jobs_uniform()
    generate_jobs_normal()
