import pandas as pd
import numpy as np
import os
import pickle
import itertools

pd.set_option('display.max_columns', None)  # or 1000
pd.set_option('display.max_rows', None)  # or 1000
pd.set_option('display.max_colwidth', None)


group_on = ['n', 'd', 'effective_variance', 'min_points', 'small field limit', 'nr of node points',
            'effective variance limit']
meanstd = ['relative error 2','abs error','time (s)']

def load_exotic_results(string_search,folder):
    files = os.listdir(folder)
    d=3
    big_data = []
    for f in files:
        mini_dat = []
        if string_search in f:
            with open(folder+f, 'rb') as s:
                if '1e6' in f:
                    N=1000000
                if '1e7' in f:
                    N=10000000
                if '1e8' in f:
                    N=100000000
                if '1e9' in f:
                    N=1000000000

                data = pickle.load(s)
                eff_var_data = 1/(data['sqls']**2 * 2)
                all_args = np.array([el for el in itertools.product(*data['kwargs_list'].values())])
                mini_dat = np.concatenate([np.ones_like(data['rel_err'][:,np.newaxis])*N,
                                           np.ones_like(data['rel_err'][:,np.newaxis])*d,np.ones_like(data['rel_err'][:,np.newaxis])*eff_var_data,all_args,data['abs_err'][:,np.newaxis],data['rel_err'][:,np.newaxis],data['elapsed'][:,np.newaxis],
                                           ],axis=1)
                big_data.append(mini_dat)
    big_data = np.concatenate(big_data,axis=0)
    df = pd.DataFrame(big_data,columns=['n', 'd', 'effective_variance','nr of node points','effective variance limit', 'min_points','var comp', 'small field limit','abs error','relative error 2', 'time (s)'])
    df.to_csv(f'{string_search}.csv')

def load_and_concat_df(exp_folder):
    files = os.listdir(exp_folder)
    df_list = []
    for f in files:
        mini_df = pd.read_csv(exp_folder+'/'+f,index_col=0)
        df_list.append(mini_df)
    return pd.concat(df_list)

def process_df(df_1):
    mean = df_1.groupby(group_on)[meanstd].mean()
    mean= mean.reset_index()
    std = df_1.groupby(group_on)[meanstd].std()
    std=std.reset_index()
    mean['relative error 2 std'] = std['relative error 2']
    mean['time (s) std'] = std['time (s)']
    return mean

if __name__ == '__main__':
    load_exotic_results('Clustered',"./benchs_sparse/results/")
    load_exotic_results('Brownian_Motion',"./benchs_sparse/results/")
    load_exotic_results('Fractional_Brownian_Motion',"./benchs_sparse/results/")
    pass
    for i in range(1,4):
        df_1 = load_and_concat_df(f'experiment_{i}')
        df_1_processsed = process_df(df_1)
        df_1_processsed.to_csv(f"experiment_{i}_results_summary.csv")
    for i in range(1,3):
        df_1 = load_and_concat_df(f'experiment_{i}_27')
        df_1_processsed = process_df(df_1)
        df_1_processsed.to_csv(f"experiment_{i}_27_results_summary.csv")

