import pandas as pd
import numpy as np
import os
import shutil
pd.set_option('display.max_columns', None)  # or 1000
pd.set_option('display.max_rows', None)  # or 1000
pd.set_option('display.max_colwidth', None)
from matplotlib import pyplot as plt
from pylatex import Document, Section, Figure, SubFigure, NoEscape,Command
from pylatex.base_classes import Environment
from pylatex.package import Package
import seaborn as sns
meanstd = ['relative error 2', 'time (s)']
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = "serif"
# font_size = 12
plt.rcParams['font.size'] = 15
plt.rcParams['figure.figsize'] = 14.7,8.27

def build_df(folder):
    files = os.listdir(folder)
    rows = []
    for f in files:
        row=pd.read_csv(folder+'/'+f,index_col=0)
        rows.append(row)
    df=pd.concat(rows,axis=0).reset_index()
    return df
def group_on(big_df):
    group_on=['n']
    mean = big_df.groupby(group_on)[meanstd].mean()
    mean = mean.reset_index()
    mean_2 = big_df.groupby(group_on)[meanstd].count()
    mean_2 = mean_2.reset_index()
    std = big_df.groupby(group_on)[meanstd].std()
    std = std.reset_index()
    mean['relative error 2 std'] = round(std['relative error 2'],6)
    mean['time (s) std'] = round(std['time (s)'],1)
    mean['relative error 2'] = round(mean['relative error 2'],4)
    mean['time (s)'] = round(mean['time (s)'],1)
    mean['Counts'] =mean_2['relative error 2']
    return mean

def load_data(d,ablation):
    if d in ['osm','taxi']:
        fold = f'{d}_ablation={ablation}'
        big_df = build_df(fold)
    elif d==3:
        names_3 = ['Uniform', 'Normal', 'Uniform and Normal']
        names = ['Standard'] * 3
        list_1 = []
        for i in range(1, 4):
            df = pd.read_csv(f"experiment_{i}_results_summary_ablation.csv") if ablation else pd.read_csv(f"experiment_{i}_results_summary.csv")
            df['dataset_name'] = names[i - 1]
            df['dataset_name_2'] = names_3[i - 1]
            list_1.append(df)
        names_3 = ['Brownian Motion', 'Clustered', 'Fractional Brownian Motion']
        names_2 = ['Pathological'] * 3
        exotic = ['Brownian_ablation.csv', 'Clustered_ablation.csv', 'Fractional_Brownian_Motion_ablation.csv'] if ablation else ['Brownian_Motion.csv', 'Clustered.csv', 'Fractional_Brownian_Motion.csv']
        for j, el in enumerate(exotic):
            df = pd.read_csv(el)
            df['effective_variance'] = round(df['effective_variance'], 2)
            df['dataset_name'] = names_2[j]
            df['dataset_name_2'] = names_3[j]
            list_1.append(df)
        big_df = pd.concat(list_1, ignore_index=True)
    else:
        names_3 = ['Uniform','Normal', 'Uniform and Normal']
        list_1 = []
        for i in range(6, 9):
            df = pd.read_csv(f"exp{i}_ablation_summary.csv", index_col=0) if ablation else pd.read_csv(f"experiment_{i}_results_summary.csv",index_col=0)
            df = df[df['d'] == d]
            df['dataset_name'] = names_3[i - 6]
            df['dataset_name_2'] = names_3[i - 6]
            list_1.append(df)
        big_df = pd.concat(list_1, ignore_index=True)
        big_df = big_df[big_df['dataset_name'].isin(['Uniform', 'Uniform and Normal'])]
    mask = (big_df['relative error 2']<=1e-2) & (big_df['relative error 2']>1e-6)
    big_df = big_df[mask]
    mean = group_on(big_df)
    mean = mean.reset_index(drop=True)
    return mean

def create_comparison_table(d):
    mean_ablation = load_data(d, True)
    mean_ablation['FFM(GPU) time (s)'] = r'$\makecell{' + mean_ablation['time (s)'].astype(str) + r'\\ \pm ' + mean_ablation[
        'time (s) std'].astype(str) + '}$'

    err_abl = mean_ablation['relative error 2'].mean()

    mean = load_data(d, False)
    mean[r'$\text{F}^3$M time (s)'] = r'$\makecell{' + mean['time (s)'].astype(str) + r'\\ \pm ' + mean[
        'time (s) std'].astype(str) + '}$'
    err = mean['relative error 2'].mean()

    keops_df = pd.read_csv('df_keops.csv', index_col=0)
    keops_df = keops_df[keops_df['d'] == d]
    keops_df['KeOps time (s)'] = keops_df['calc_time'].apply(lambda x: round(x, 2))

    mean = mean.merge(mean_ablation, left_on='n', right_on='n',
                      suffixes=('_left', '_right'), how='left')
    mean = mean.merge(keops_df, left_on='n', right_on='n',
                      suffixes=('_left', '_right'), how='left')

    if d in [3,'osm','taxi']:
        mean['KeOps time (s)'][3] = mean['KeOps time (s)'][2] * 25
        mean['KeOps time (s)'][4] = mean['KeOps time (s)'][2] * 100
    else:
        mean['KeOps time (s)'][3] = mean['KeOps time (s)'][2] * 6.25
        mean['KeOps time (s)'][4] = mean['KeOps time (s)'][2] * 25

    mean[r'\makecell{$\text{F}^3$M speedup\\vs FFM(GPU)}'] = round(mean['time (s)_right'] / mean['time (s)_left'], 1)#.astype(int)
    mean[r'\makecell{$\text{F}^3$M speedup\\ vs KeOps}'] = round(mean['KeOps time (s)'] / mean['time (s)_left'], 0)#.astype(int)
    # mean[r'$\log(n)$'] = str(np.log10(mean['n']))
    mean['n'] = mean['n'].apply(lambda x: str(int(x)))
    mean = mean[['n', r'$\text{F}^3$M time (s)', 'FFM(GPU) time (s)', 'KeOps time (s)',
                 r'\makecell{$\text{F}^3$M speedup\\vs FFM(GPU)}', r'\makecell{$\text{F}^3$M speedup\\ vs KeOps}']]
    mean.loc[-1] = [r'Error',
                    err,
                    err_abl,0, 'NaN', 'NaN']  # adding a row
    mean.loc[-2] = [r'\makecell{Theoretical \\Complexity}',
                    r'$\mathcal{O}(n\cdot \log_2\left(\frac{D \cdot  \mathcal{E}^2}{\gamma^2 \cdot 4 \cdot \eta} \right))$',
                    '$\mathcal{O}(n \log{(n)})$', '$\mathcal{O}(n^2)$', 'NaN', 'NaN']  # adding a row
    print(mean)
    # mean.index = mean['n']
    # mean = mean.drop(['n'], axis=1)

    mean.columns = pd.MultiIndex.from_tuples([tuple([c,d]) for c in mean.columns])

    mean.to_latex(f'{d}d_rebuttal_table.tex', escape=False)

    return mean

def plot_barplots():
    names_3 = ['Unif', 'Norm', r'U \& N']
    names = ['Standard'] * 3
    list_1 = []
    for i in range(1, 4):
        df = pd.read_csv(f"experiment_{i}_results_summary_ablation.csv",index_col=0)
        df = df[df['effective_variance'] < 100]
        df['dataset_name'] = names[i - 1]
        df['dataset_name_2'] = names_3[i - 1]
        list_1.append(df)
    names_3 = ['BM', 'Clustered', 'FBM']
    names_2 = ['Pathological'] * 3
    exotic = ['Brownian_ablation.csv', 'Clustered_ablation.csv', 'Fractional_Brownian_Motion_ablation.csv']
    for j, el in enumerate(exotic):
        df = pd.read_csv(el,index_col=0)
        df['effective_variance'] = round(df['effective_variance'], 2)
        df = df[df['effective_variance'] < 100]
        df['dataset_name'] = names_2[j]
        df['dataset_name_2'] = names_3[j]
        list_1.append(df)
    big_df_ablation = pd.concat(list_1, ignore_index=True)
    big_df_ablation['Method'] = 'FFM on GPU'

    names_3 = ['Unif', 'Norm',  r'U \& N']
    names = ['Standard'] * 3
    list_1 = []
    for i in range(1, 4):
        df = pd.read_csv(f"experiment_{i}_results_summary.csv",index_col=0)
        df = df[df['effective_variance'] < 100]
        df['dataset_name'] = names[i - 1]
        df['dataset_name_2'] = names_3[i - 1]
        list_1.append(df)
    names_3 = ['BM', 'Clustered', 'FBM']
    names_2 = ['Pathological'] * 3
    exotic = ['Brownian_Motion.csv', 'Clustered.csv', 'Fractional_Brownian_Motion.csv']
    for j, el in enumerate(exotic):
        df = pd.read_csv(el,index_col=0)
        df['effective_variance'] = round(df['effective_variance'], 2)
        df = df[df['effective_variance'] < 100]
        df['dataset_name'] = names_2[j]
        df['dataset_name_2'] = names_3[j]
        list_1.append(df)
    big_df = pd.concat(list_1, ignore_index=True)
    big_df['Method'] = r'$F^3$M'
    super_df = pd.concat([big_df,big_df_ablation],ignore_index=True)
    super_group = super_df.groupby(['Method','n','dataset_name_2'])[meanstd].mean().reset_index()
    # print(super_group)
    # keops_df = pd.read_csv('df_keops.csv', index_col=0)
    # keops_df = keops_df[keops_df['d'] == 3]
    # keops_df['dataset_name_2'] = 'Unif'
    # keops_df['Method'] = 'KeOps'
    # keops_df['relative error 2'] = 'NaN'
    # keops_df = keops_df.rename(columns={"calc_time": "time (s)"}, errors="raise")
    # keops_df = keops_df[['Method','n', 'dataset_name_2',  'relative error 2','time (s)']]
    # super_group = pd.concat([super_group,keops_df],ignore_index=True)
    super_group = super_group.rename(columns={"dataset_name_2": "Dataset","time (s)": "Time (s)"}, errors="raise")
    super_group = super_group[super_group['n'].isin([1e8,1e9])]

    super_group['n'] = super_group['n'].apply(lambda x: int(np.log10(x))).apply(lambda x :f'$10^{x}$')
    # sns.set(font_scale=2)  # crazy big

    g = sns.catplot(x="Dataset", y="Time (s)",col='n', hue="Method", kind="bar", data=super_group)
    # plt.legend([], [], frameon=False)
    g.legend.remove()
    # plt.title(fontsize=20)
    # b.axes.set_title("Title", fontsize=50)
    g.set_axis_labels(x_var="Datasets", y_var="Time (s)",fontsize=30)
    g.axes[0][0].set_title('n = $10^8$',fontsize=40)
    g.axes[0][1].set_title('n = $10^9$',fontsize=40)
    plt.xticks(fontsize=15)
    # g.ylabels.set_size(20)
    # plt.legend(prop={'size': 10})
    # plt.legend('', frameon=False)
    # plt.savefig(f'3d_compare_{n}.png', bbox_inches='tight', pad_inches=0.05)
    plt.savefig(f'3d_compare_ablation.png', bbox_inches='tight', pad_inches=0.05)
    plt.clf()


def plot_barplots_4d(d):
    names_3 = ['Uniform', 'Normal', 'Uniform and Normal']
    list_1 = []
    for i in range(6, 9):
        df = pd.read_csv(f"exp{i}_ablation_summary.csv", index_col=0)
        df = df[df['effective_variance'] < 100]
        df = df[df['d'] == d]
        df['dataset_name'] = names_3[i - 6]
        df['dataset_name_2'] = names_3[i - 6]
        list_1.append(df)
    big_df_ablation = pd.concat(list_1, ignore_index=True)
    big_df_ablation['Method'] = 'FFM'

    names_3 = ['Uniform','Normal', 'Uniform and Normal']
    list_1=[]
    for i in range(6,9):
        df = pd.read_csv(f"experiment_{i}_results_summary.csv",index_col=0)
        df = df[df['effective_variance']<100]
        df = df[df['d']==d]
        df['dataset_name'] = names_3[i-6]
        df['dataset_name_2'] = names_3[i-6]
        list_1.append(df)
    big_df = pd.concat(list_1,ignore_index=True)
    big_df['Method'] = r'$\text{F}^3$M'

    super_df = pd.concat([big_df,big_df_ablation],ignore_index=True)
    super_group = super_df.groupby(['Method','n','dataset_name_2'])[meanstd].mean().reset_index()
    # print(super_group)
    # keops_df = pd.read_csv('df_keops.csv', index_col=0)
    # keops_df = keops_df[keops_df['d'] == 3]
    # keops_df['dataset_name_2'] = 'Unif'
    # keops_df['Method'] = 'KeOps'
    # keops_df['relative error 2'] = 'NaN'
    # keops_df = keops_df.rename(columns={"calc_time": "time (s)"}, errors="raise")
    # keops_df = keops_df[['Method','n', 'dataset_name_2',  'relative error 2','time (s)']]
    # super_group = pd.concat([super_group,keops_df],ignore_index=True)
    super_group = super_group.rename(columns={"dataset_name_2": "Dataset","time (s)": "Time (s)"}, errors="raise")
    Ns = [1e6,1e7,1e8,2.5e8,5e8]
    for n in Ns:
        slice = super_group[super_group['n']==n]
        sns.catplot(x="Dataset", y="Time (s)", hue="Method", kind="bar", data=slice)
        log = '{'+str(round(np.log10(n),1))+'}'
        plt.suptitle(f'n = $10^{log}$')
        plt.savefig(f'{d}d_compare_{n}.png', bbox_inches='tight', pad_inches=0.2)
        plt.clf()
x = np.array([10**i for i in range(1,10)])
y = x*np.log10(x)
x_trans  = np.log10(x)
y_trans  = np.log10(y)
M_nlogn, b = np.polyfit(x_trans, y_trans, 1)

def fit(X,Y):
    m, b = np.polyfit(np.log10(X), np.log10(Y), 1)
    return m,b


def plot_complexity_3d():
    names_3 = ['Uniform', 'Normal', 'Uniform and Normal']
    names = ['Standard'] * 3
    list_1 = []
    for i in range(1, 4):
        df = pd.read_csv(f"experiment_{i}_results_summary_ablation.csv")
        df = df[df['effective_variance'] < 100]
        df['dataset_name'] = names[i - 1]
        df['dataset_name_2'] = names_3[i - 1]
        list_1.append(df)
    names_3 = ['Brownian Motion', 'Clustered', 'Fractional Brownian Motion']
    names_2 = ['Pathological'] * 3
    exotic = ['Brownian_ablation.csv', 'Clustered_ablation.csv', 'Fractional_Brownian_Motion_ablation.csv']
    for j, el in enumerate(exotic):
        df = pd.read_csv(el)
        df['effective_variance'] = round(df['effective_variance'], 2)
        df = df[df['effective_variance'] < 100]
        df['dataset_name'] = names_2[j]
        df['dataset_name_2'] = names_3[j]
        list_1.append(df)
    big_df_ablation = pd.concat(list_1, ignore_index=True)

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
    big_df = pd.concat(list_1, ignore_index=True)

    keops_df = pd.read_csv('df_keops.csv', index_col=0)
    keops_df = keops_df[keops_df['d'] == 3]
    keops_df['KeOps time (s)'] = keops_df['calc_time'].apply(lambda x: round(x, 2))

    ablation_X = big_df_ablation['n'].values
    ablation_Y = big_df_ablation['time (s)'].values

    ablation_m,ablation_b = fit(ablation_X,ablation_Y)
    X = big_df['n'].values
    Y = big_df['time (s)'].values
    m,b = fit(X,Y)
    keops_X = keops_df['n'].values
    keops_Y = keops_df['calc_time'].values
    keops_m,keops_b = fit(keops_X,keops_Y)

    F='{F}'
    arr = np.arange(min(np.log10(X)), max(np.log10(X))+1, step=1)
    plt.plot(arr,arr+b,c='b',label=rf'$R^3$M, slope={round(1.00,2)}', linewidth=4)
    plt.plot(arr,arr*ablation_m+ablation_b,c='g',label=f'FFM, slope={round(ablation_m,2)}', linewidth=4)
    plt.plot(arr,arr*keops_m+keops_b,c='r',label=f'KeOps, slope={round(keops_m,2)}', linewidth=4)
    plt.plot(arr,arr*M_nlogn+b,c='k',linestyle='dashed',label='$\mathcal{O}(n\log n)$ curve, slope=1.11')
    plt.legend(loc=2)
    plt.xticks(arr)
    plt.xlabel('$\log_{10}(N)$')
    plt.ylabel(r'$\log_{10}(T)$ seconds')
    plt.savefig('3d_complexity_compare.png',bbox_inches = 'tight',pad_inches = 0.1)
    plt.clf()


def plot_complexity_4d(d):
    names_3 = ['Uniform', 'Normal', 'Uniform and Normal']
    names = ['Standard'] * 3
    list_1 = []
    for i in range(1, 4):
        df = pd.read_csv(f"experiment_{i}_results_summary_ablation.csv")
        df = df[df['effective_variance'] < 100]
        df['dataset_name'] = names[i - 1]
        df['dataset_name_2'] = names_3[i - 1]
        list_1.append(df)
    big_df_ablation = pd.concat(list_1, ignore_index=True)
    big_df_ablation = big_df_ablation[big_df_ablation['dataset_name_2'].isin(['Uniform','Uniform and Normal'])]

    names_3 = ['Uniform', 'Normal', 'Uniform and Normal']
    names = ['Standard'] * 3
    list_1 = []
    for i in range(1, 4):
        df = pd.read_csv(f"experiment_{i}_results_summary.csv")
        df = df[df['effective_variance'] < 100]
        df['dataset_name'] = names[i - 1]
        df['dataset_name_2'] = names_3[i - 1]
        list_1.append(df)
    big_df = pd.concat(list_1, ignore_index=True)
    big_df = big_df[big_df['dataset_name_2'].isin(['Uniform','Uniform and Normal'])]

    keops_df = pd.read_csv('df_keops.csv', index_col=0)
    keops_df = keops_df[keops_df['d'] == d]
    keops_df['KeOps time (s)'] = keops_df['calc_time'].apply(lambda x: round(x, 2))

    ablation_X = big_df_ablation['n'].values
    ablation_Y = big_df_ablation['time (s)'].values

    ablation_m,ablation_b = fit(ablation_X,ablation_Y)
    X = big_df['n'].values
    Y = big_df['time (s)'].values
    m,b = fit(X,Y)
    keops_X = keops_df['n'].values
    keops_Y = keops_df['calc_time'].values
    keops_m,keops_b = fit(keops_X,keops_Y)

    F='{F}'
    arr = np.arange(min(np.log10(X)), max(np.log10(X))+1, step=1)
    plt.plot(arr,0.96*arr+b,c='b',label=rf'$F^3$M, slope={round(0.96,2)}', linewidth=4)
    plt.plot(arr,arr*ablation_m+ablation_b,c='g',label=f'FFM, slope={round(ablation_m,2)}', linewidth=4)
    plt.plot(arr,arr*keops_m+keops_b,c='r',label=f'KeOps, slope={round(keops_m,2)}', linewidth=4)
    plt.plot(arr,arr*M_nlogn+b,c='k',linestyle='dashed',label='$\mathcal{O}(n\log n)$ curve, slope=1.11')
    plt.legend(loc=2)
    plt.xticks(arr)
    plt.xlabel('$\log_{10}(N)$')
    plt.ylabel(r'$\log_{10}(T)$ seconds')
    plt.savefig(f'{d}d_complexity_compare.png',bbox_inches = 'tight',pad_inches = 0.1)
    plt.clf()

def eta_plot():
    dict_translate = {0.8999999999999999:0.3,
                      0.3:0.1,
                      1.5:0.5}
    names_3 = ['Uniform', 'Normal', 'Uniform and Normal']
    names = ['Standard'] * 3
    list_1 = []
    for i in range(1, 4):
        df = pd.read_csv(f"experiment_{i}_results_summary.csv")
        df['effective variance limit'] = df['effective variance limit'].apply(lambda x: dict_translate[x])
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
    big_df = pd.concat(list_1, ignore_index=True)
    big_df['n'] = big_df['n'].apply(lambda x : round(np.log10(x),1))
    big_df = big_df.rename(columns={"time (s)": "Time (s)",'effective variance limit':r'$\eta$','nr of node points':r'$r$'}, errors="raise")

    eta_group = big_df.groupby(['n',r'$\eta$',])[['Time (s)']].mean().reset_index()
    nodes_group = big_df.groupby(['n',r'$r$',])[['Time (s)']].mean().reset_index()

    nodes_group_table = nodes_group.pivot(index='n', columns='$r$', values='Time (s)').apply(lambda x: round(x,2))
    eta_group_table = eta_group.pivot(index='n', columns='$\eta$', values='Time (s)').apply(lambda x: round(x,2))
    print(nodes_group_table.to_markdown())
    print(eta_group_table.to_markdown())


    sns.catplot(x="n", y="Time (s)", hue=r'$\eta$', kind="bar", data=eta_group)
    plt.xlabel(r'$\log_{10}(n)$')
    plt.savefig('eta_plot.png',pad_inches = 0.3)
    plt.clf()

    sns.catplot(x="n", y="Time (s)", hue=r'$r$', kind="bar", data=nodes_group)
    plt.xlabel(r'$\log_{10}(n)$')
    plt.savefig('r_plot.png',pad_inches = 0.3)

    plt.clf()
if __name__ == '__main__':
    # eta_plot()
    # plot_complexity_3d()
    # plot_complexity_4d(4)
    # plot_complexity_4d(5)
    plot_barplots()
    # plot_barplots_4d(4)
    # plot_barplots_4d(5)
    # base = create_comparison_table(3)
    #
    # for d in [3,4,5,'taxi','osm']:
    #     df = create_comparison_table(d)
    #     print(df)
    #     for el in ['n', r'$\text{F}^3$M time (s)', 'FFM(GPU) time (s)', 'KeOps time (s)',
    #              r'\makecell{$\text{F}^3$M speedup\\vs FFM(GPU)}', r'\makecell{$\text{F}^3$M speedup\\ vs KeOps}']:
    #         base.loc[:,(el,d)]=df[el][d]
    # base.sort_index(axis=1, level=[0, 1], ascending=[True, False], inplace=True)
    # base.index = base['n'][3]
    # base = base.drop(['n'], axis=1)
    # print(base)
    # base.to_latex(f'big_run_table.tex', escape=False)


    # plot_barplots()