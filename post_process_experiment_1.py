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


def load_exotic_results(string_search,folder,save_name):
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
    df.to_csv(f'{save_name}.csv')

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
    # load_exotic_results('Clustered',"./benchs_sparse/results/",'Clustered')
    # load_exotic_results('Brownian_Motion',"./benchs_sparse/results/",'Brownian_Motion')
    # load_exotic_results('Fractional_Brownian_Motion',"./benchs_sparse/results/",'Fractional_Brownian_Motion')
    # for i in range(1,4):
    #     # df_1 = load_and_concat_df(f'old_experiments_3/experiment_{i}')
    #     df_1 = load_and_concat_df(f'experiment_{i}_aggresive')
    #     df_1_processsed = process_df(df_1)
    #     df_1_processsed.to_csv(f"experiment_{i}_results_summary.csv")
    #
    #
    # load_exotic_results('Clustered',"./benchs_sparse/res_ablation/",'Clustered_ablation')
    # load_exotic_results('Brownian_Motion',"./benchs_sparse/res_ablation/",'Brownian_ablation')
    # load_exotic_results('Fractional_Brownian_Motion',"./benchs_sparse/res_ablation/",'Fractional_Brownian_Motion_ablation')
    #
    # for i in range(1,4):
    #     df_1 = load_and_concat_df(f'experiment_{i}_ablation')
    #     df_1_processsed = process_df(df_1)
    #     df_1_processsed.to_csv(f"experiment_{i}_results_summary_ablation.csv")
    #
    # for i in range(6,9):
    #     df_1 = load_and_concat_df(f'exp{i}_ablation_2')
    #     df_1_processsed = process_df(df_1)
    #     df_1_processsed.to_csv(f"exp{i}_ablation_summary.csv")
    #
    # df_keops = load_and_concat_df('keops_ref_bench')
    # df_keops.to_csv('df_keops.csv')
    # for i in range(6,9):
    #     df_1 = load_and_concat_df(f'experiment_{i}_new')
    #     df_1_processsed = process_df(df_1)
    #     df_1_processsed.to_csv(f"experiment_{i}_results_summary.csv")
    #     list_cat =[]

    list_cat = []
    for i in range(1,4):
        df_1 = load_and_concat_df(f'experiment_{i}_78D_uniform')
        df_1_processsed = process_df(df_1)
        df_1_processsed.to_csv(f"experiment_{i}_78D_uniform.csv")

    # for i in range(1,4):
    #     df_1 = load_and_concat_df(f'experiment_10_{i}')
    #     df_1_processsed = process_df(df_1)
    #     list_cat.append(df_1_processsed)
    # big_list = pd.concat(list_cat)
    # big_list.to_csv(f"experiment_10_results_summary.csv")
