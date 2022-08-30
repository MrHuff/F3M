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
    # df = pd.DataFrame(big_data,columns=['n', 'd', 'effective_variance','nr of node points','effective variance limit', 'min_points','var comp', 'small field limit','abs error','relative error 2', 'time (s)'])
    df = pd.DataFrame(big_data,columns=['n', 'd', 'effective_variance','nr of node points','effective variance limit', 'min_points','var comp','abs error','relative error 2', 'time (s)'])
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


mode_ref = {3:'Uniform and Normal',2:'Normal',1:'Uniform'}

def process_df_2(job_folder,results_folder):
    all_job_paths = []
    for el in os.listdir(job_folder):
        for el_2 in os.listdir(job_folder + '/' + el):
            all_job_paths.append(job_folder + '/' + el +'/'+el_2)

    data = []
    for j in all_job_paths:
        d = pickle.load(open(j, "rb"))
        index = d['counter']
        result_path = f'{results_folder}/{results_folder}_{index}.csv'
        if os.path.exists(result_path):
            row = pd.read_csv(result_path)
            data.append([mode_ref[d['mode']],d['n'],d['d'],d['nr_of_interpolation'], d['small_field_limit'],row['effective_variance'].values[0],row['effective variance limit'].values[0],row['relative error 2'].values[0],row['time (s)'].values[0]])
    all_df = pd.DataFrame(data,columns=['dataset_name','n','d','nr of node points','small field limit','effective_variance','effective variance limit','relative error 2','time (s)']).reset_index(drop=True)
    all_df.to_csv(f'{results_folder}.csv')
    return all_df
if __name__ == '__main__':
    df_2 = process_df_2('3d_jobs_25','3d_jobs_25_results')

    load_exotic_results('Clustered',"./benchs_sparse/res_ablation_25/",'Clustered_ablation_25')
    load_exotic_results('Brownian_Motion',"./benchs_sparse/res_ablation_25/",'Brownian_Motion_25')
    load_exotic_results('Fractional_Brownian_Motion',"./benchs_sparse/res_ablation_25/",'Fractional_Brownian_Motion_25')
    # for i in range(1,4):
    #     df_1 = load_and_concat_df(f'experiment_{i}_aggresive')
    #     df_1_processsed = process_df(df_1)
    #     df_1_processsed.to_csv(f"experiment_{i}_results_summary_25.csv")
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


    # for i in range(1,4):
    #     df_1 = load_and_concat_df(f'experiment_10_{i}')
    #     df_1_processsed = process_df(df_1)
    #     list_cat.append(df_1_processsed)
    # big_list = pd.concat(list_cat)
    # big_list.to_csv(f"experiment_10_results_summary.csv")
#
# load_these = ['experiment_6_5',
#               'experiment_6_256_rest',
#               'experiment_6_512_10',
#               'experiment_6_512_rest',
#               'experiment_6_1024_10',
#               'experiment_6_1024_rest',
#               'experiment_7_5',
#               'experiment_8_hack'
#               ]
# dataset_name = ['Uniform']*6+['Normal']+['Uniform and Normal']
#
# concated_df=[]
#
# for el,name in zip(load_these,dataset_name):
#     df_1 = load_and_concat_df(el)
#     df_1_processsed = process_df(df_1)
#     df_1_processsed['dataset_name'] = name
#     df_1_processsed['dataset_name_2'] = '5D'
#     concated_df.append(df_1_processsed)
# big_df = pd.concat(concated_df)
# big_df.to_csv("5d_exp.csv")
#
# pd.set_option('display.float_format', '{:.2E}'.format)

# names_3 = ['Uniform', 'Normal', 'Uniform and Normal']
# names = ['Standard'] * 3
# list_1 = []
# for i in range(1, 4):
#     df = pd.read_csv(f"experiment_{i}_results_summary.csv")
#     df = df[df['effective_variance'] < 100]
#     df['dataset_name'] = names[i - 1]
#     df['dataset_name_2'] = names_3[i - 1]
#     list_1.append(df)
# names_3 = ['Brownian Motion', 'Clustered', 'Fractional Brownian Motion']
# names_2 = ['Pathological'] * 3
# exotic = ['Brownian_Motion.csv', 'Clustered.csv', 'Fractional_Brownian_Motion.csv']
# for j, el in enumerate(exotic):
#     df = pd.read_csv(el)
#     df['effective_variance'] = round(df['effective_variance'], 2)
#     df = df[df['effective_variance'] < 100]
#     df['dataset_name'] = names_2[j]
#     df['dataset_name_2'] = names_3[j]
#     list_1.append(df)
#
# group_on=['n']
# big_df = pd.concat(list_1, ignore_index=True)
#
# mean = big_df.groupby(group_on)[meanstd].mean()
# mean = mean.reset_index()
# std = big_df.groupby(group_on)[meanstd].std()
# std = std.reset_index()
# mean['relative error 2 std'] = round(std['relative error 2'],3)
# mean['time (s) std'] = round(std['time (s)'],2)
# mean['relative error 2'] = round(mean['relative error 2'],4)
# mean['time (s)'] = round(mean['time (s)'],2)
# print(mean)
# df = pd.read_csv("experiment_10_results_summary.csv",index_col=0)
# df = df[df['effective variance limit']==0.5]
# df = df[df['nr of node points']==64]
# group_on=['n']
# mean = df.groupby(group_on)[meanstd].mean()
# mean = mean.reset_index()
# std = df.groupby(group_on)[meanstd].std()
# std = std.reset_index()
# mean['relative error 2 std'] = round(std['relative error 2'],5)
# mean['time (s) std'] = round(std['time (s)'],2)
# mean['relative error 2'] = round(mean['relative error 2'],4)
# mean['time (s)'] = round(mean['time (s)'],2)
# print(mean)