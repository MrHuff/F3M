from post_process_experiment_1 import *
#
#
# load_these = ['experiment_6_5',
#               'experiment_6_256_rest',
#               'experiment_6_512_10',
#               'experiment_6_512_rest',
#               'experiment_6_1024_10',
#               'experiment_6_1024_rest',
#               'experiment_7_5',
#               'experiment_8'
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

names_3 = ['Uniform', 'Normal', 'Uniform and Normal']
names = ['Standard'] * 3
list_1 = []
for i in range(1, 4):
    df = pd.read_csv(f"experiment_{i}_results_summary.csv")
    df = df[df['effective_variance'] < 100]
    df['dataset_name'] = names[i - 1]
    df['dataset_name_2'] = names_3[i - 1]
    list_1.append(df)
names_3 = ['Brownian Motion', 'Clustered', 'Fractional Brownian Motion']
names_2 = ['Pathological'] * 3
exotic = ['Brownian_Motion.csv', 'Clustered.csv', 'Fractional_Brownian_Motion.csv']
for j, el in enumerate(exotic):
    df = pd.read_csv(el)
    df['effective_variance'] = round(df['effective_variance'], 2)
    df = df[df['effective_variance'] < 100]
    df['dataset_name'] = names_2[j]
    df['dataset_name_2'] = names_3[j]
    list_1.append(df)

group_on=['n']
big_df = pd.concat(list_1, ignore_index=True)

mean = big_df.groupby(group_on)[meanstd].mean()
mean = mean.reset_index()
std = big_df.groupby(group_on)[meanstd].std()
std = std.reset_index()
mean['relative error 2 std'] = round(std['relative error 2'],3)
mean['time (s) std'] = round(std['time (s)'],2)
mean['relative error 2'] = round(mean['relative error 2'],4)
mean['time (s)'] = round(mean['time (s)'],2)
print(mean)