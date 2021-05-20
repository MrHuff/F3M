import pandas as pd
import numpy as np
import os
pd.set_option('display.max_columns', None)  # or 1000
pd.set_option('display.max_rows', None)  # or 1000
pd.set_option('display.max_colwidth', None)


group_on = ['n', 'd', 'effective_variance', 'min_points', 'small field limit', 'nr of node points',
            'effective variance limit']
meanstd = ['relative error 2', 'time (s)']

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
    pass
    # for i in range(1,10):
    #     df_1 = load_and_concat_df(f'current_exp_1/experiment_{i}')
    #     df_1_processsed = process_df(df_1)
    #     df_1_processsed.to_csv(f"experiment_{i}_results_summary.csv")
    # i=5
    # df_1 = load_and_concat_df(f'current_exp_1/experiment_{i}')
    # print(df_1)



