import pandas as pd
import numpy as np
import os
pd.set_option('display.max_columns', None)  # or 1000
pd.set_option('display.max_rows', None)  # or 1000
pd.set_option('display.max_colwidth', None)
from matplotlib import pyplot as plt

group_on = ['n', 'd', 'effective_variance', 'min_points', 'small field limit', 'nr of node points',
            'effective variance limit']
meanstd = ['relative error 2', 'time (s)']


if __name__ == '__main__':
    pass
    # for i in range(1,10):
    #     df_1 = load_and_concat_df(f'current_exp_1/experiment_{i}')
    #     df_1_processsed = process_df(df_1)
    #     df_1_processsed.to_csv(f"experiment_{i}_results_summary.csv")
    # i=5
    # df_1 = load_and_concat_df(f'current_exp_1/experiment_{i}')
    # print(df_1)



