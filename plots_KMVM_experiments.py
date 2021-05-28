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

class subfigure(Environment):
    """A class to wrap LaTeX's alltt environment."""
    packages = [Package('subcaption')]
    escape = False
    content_separator = "\n"
    _repr_attributes_mapping = {
        'position': 'options',
        'width': 'arguments',
    }

    def __init__(self, position=NoEscape(r'H'),width=NoEscape(r'0.45\linewidth'), **kwargs):
        """
        Args
        ----
        width: str
            Width of the subfigure itself. It needs a width because it is
            inside another figure.
        """

        super().__init__(options=position,arguments=width, **kwargs)


x=20
plt.rcParams['figure.figsize'] = 1.4*x, x

font_size = 80
plt.rcParams['font.size'] = font_size
plt.rcParams['legend.fontsize'] = 80
plt.rcParams['axes.labelsize'] = font_size
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = "serif"

group_on = ['n', 'd', 'effective_variance', 'min_points', 'small field limit', 'nr of node points',
            'effective variance limit']
meanstd = ['relative error 2', 'time (s)']


x = np.array([10**i for i in range(1,10)])
y = x*np.log10(x)

x_trans  = np.log10(x)
y_trans  = np.log10(y)
# plt.plot(x_trans,y_trans)
# plt.show()

M_nlogn, b = np.polyfit(x_trans, y_trans, 1)
print(round(M_nlogn,2))

def complexity_plots(savefig, df, X, Y, slice,label_nice):
    fig, ax = plt.subplots()
    prop = ax._get_lines.prop_cycler
    mean = Y
    els = df[slice].unique().tolist()

    big_X,big_Y = df[X],df[Y]

    for label_df in els:
        color = next(prop)['color']
        sub_df = df[df[slice]==label_df]
        m, b = np.polyfit(np.log10(sub_df[X]).astype(int), np.log10(sub_df[mean]), 1)
        print(m, b)
        tmp_var = round(label_df,2) if isinstance(label_df,float) else label_df
        plt.scatter(np.log10(sub_df[X]), np.log10(sub_df[mean]), marker='o',alpha=0.7,color=color,s=300,label=f'{label_nice}: {tmp_var}')
        # plt.plot(np.arange(min(np.log10(big_X)), max(np.log10(big_X)) + 1, step=1),
        #          np.arange(min(np.log10(big_X)), max(np.log10(big_X)) + 1, step=1) * m + b,color=color)
        # plt.plot([], [], 'o' ,label=f'{label_nice}: {round(label_df,2)}',color=color)

    m, b = np.polyfit(np.log10(big_X), np.log10(big_Y), 1)
    print(m,b)
    plt.plot(np.arange(min(np.log10(big_X)), max(np.log10(big_X))+1, step=1),np.arange(min(np.log10(big_X)), max(np.log10(big_X))+1, step=1)*m+b,c='k',label=f'Complexity curve, slope={round(m,2)}', linewidth=4)
    plt.plot(np.arange(min(np.log10(big_X)), max(np.log10(big_X))+1, step=1),np.arange(min(np.log10(big_X)), max(np.log10(big_X))+1, step=1)*M_nlogn+b,c='k',linestyle='dashed',label='$\mathcal{O}(n\log n)$ curve, slope=1.11')
    plt.legend(loc=2)
    plt.xticks(np.arange(min(np.log10(big_X)), max(np.log10(big_X))+1, step=1))

    plt.xlabel('$\log_{10}(N)$')
    plt.ylabel(r'$\log_{10}(T)$ seconds')
    plt.savefig(savefig,bbox_inches = 'tight',pad_inches = 0)
    plt.clf()


def error_plot(savefig, df, X, Y, slice,label_nice):
    mean = Y
    std = Y+' std'
    els = df[slice].unique().tolist()
    df['log_mean'] = np.log10(df[mean])
    df['log_X']  = np.log10(df[X])
    if (slice=='nr of node points'):
        df[slice] = df[slice].apply(lambda x: f'$r={int(x)}$')
    # dd = pd.melt(df, id_vars=X,var_name=slice,)
    g = sns.boxplot(x='log_X', y='log_mean', data=df, hue=slice)
    el = df['log_X'].unique().tolist()
    el = [round(e,1) for e in el]
    g.set(xticklabels=sorted(el))
    g.legend(loc=2)
    # for label_df in els:

        # sub_df = df[df[slice]==label_df]
        # un_x = sub_df[X].unique()
        # mean_2 = sub_df.groupby(X)['log_mean'].mean()
        # plt.scatter(np.log10(sub_df[X]), np.log10(sub_df[mean]), marker='o',label=f'{label_nice}: {label_df}',alpha=0.7,s=300)
        # plt.plot(sorted(np.log10(un_x)),mean_2)
    # plt.xticks(np.arange(min(np.log10(df[X])), max(np.log10(df[X]))+1, step=1))
    plt.xlabel('$\log_{10}(N)$')
    plt.ylabel(r'Relative Error $\log_{10}(x)$')
    # plt.legend(loc=0)
    plt.savefig(savefig,bbox_inches = 'tight',pad_inches = 0)
    plt.clf()


def get_worse_best(save_tex,df,X,Y):
    max = df.loc[df.groupby([X])[Y].idxmax()][['effective_variance','nr of node points',
            'effective variance limit','relative error 2', 'time (s)','dataset_name_2']]
    max.columns = ['EV','$r$','$\eta$','Relative Error','Time (s)','Dataset']
    max.columns = pd.MultiIndex.from_product([['Worst'],max.columns ])
    min = df.loc[df.groupby([X])[Y].idxmin()][['n','effective_variance','nr of node points',
            'effective variance limit','relative error 2', 'time (s)','dataset_name_2']]
    min['n'] = np.log10(min['n']).astype(int)
    min.columns = [r'$\log_{10}(N)$','EV','$r$','$\eta$','Relative Error','Time (s)','Dataset']

    min.columns = pd.MultiIndex.from_product([['Best'],min.columns ])
    max = max.reset_index(drop=True)
    min = min.reset_index(drop=True)
    df = pd.concat([min,max],axis=1)

    # df.loc[:,0]= np.log10(df['Best'][r'$\log_{10}(N)$'])
    df.to_latex(save_tex,escape=False,index=False)
    return df

def build_plot(df,plot_name):
    if not os.path.exists(plot_name):
        os.makedirs(plot_name)
    else:
        shutil.rmtree(plot_name)
        os.makedirs(plot_name)

    complexity_plots(f'{plot_name}/test_{1}.png', df, 'n', meanstd[1], 'effective variance limit','$\eta$')
    error_plot(f'{plot_name}/test_{2}.png', df, 'n', meanstd[0], 'nr of node points','Nodes') #Show independence of n
    complexity_plots(f'{plot_name}/test_{3}.png', df, 'n', meanstd[1], 'dataset_name','Group')
    error_plot(f'{plot_name}/test_{4}.png', df, 'n', meanstd[0], 'dataset_name','Group') #Show independence of n
    get_worse_best(f'{plot_name}/best_0.tex',df,'n',meanstd[0])
    get_worse_best(f'{plot_name}/worst_0.tex',df,'n',meanstd[1])

    doc = Document(default_filepath=f'{plot_name}/subfig_tex')
    string_append = ''
    with doc.create(Figure(position='H')) as plot:
        for i,p in enumerate(range(1,5)):
            p_str = f'{plot_name}/test_{p}.png'
            string_append+=r'\includegraphics[width=0.33\linewidth]{%s}'%p_str + '%\n'
            if (i+1)%2==0:
                with doc.create(subfigure(position='H', width=NoEscape(r'\linewidth'))):
                    doc.append(string_append)
                string_append=''
    doc.generate_tex()

if __name__ == '__main__':

    names_3 = ['Uniform','Normal', 'Uniform and Normal']
    names = ['Standard']*3
    list_1=[]
    for i in range(1,4):
        df = pd.read_csv(f"experiment_{i}_results_summary.csv")
        df = df[df['effective_variance']<100]
        df['dataset_name'] = names[i-1]
        df['dataset_name_2'] = names_3[i-1]
        list_1.append(df)
    names_3  = ['Brownian Motion','Clustered', 'Fractional Brownian Motion']
    names_2 = ['Pathological']*3
    exotic = ['Brownian_Motion.csv','Clustered.csv','Fractional_Brownian_Motion.csv']
    for j,el in enumerate(exotic):
        df = pd.read_csv(el)
        df['effective_variance'] = round(df['effective_variance'],2)
        df = df[df['effective_variance']<100]
        df['dataset_name'] = names_2[j]
        df['dataset_name_2'] = names_3[j]
        list_1.append(df)
    big_df = pd.concat(list_1,ignore_index=True)




    build_plot(big_df,f'plot_1')

    names_3 = ['Uniform','Normal', 'Uniform and Normal']
    names = ['4D']*3
    list_1=[]
    for i in range(6,9):
        df = pd.read_csv(f"experiment_{i}_results_summary.csv",index_col=0)
        df = df[df['effective_variance']<100]
        df = df[df['d']==4]
        df['dataset_name'] = names_3[i-6]
        df['dataset_name_2'] = names_3[i-6]
        list_1.append(df)
    big_df = pd.concat(list_1,ignore_index=True)
    print(big_df)
    build_plot(big_df,f'plot_2')

    df = pd.read_csv(f"5d_exp.csv", index_col=0)
    df = df[df['effective_variance'] < 100]
    df = df[df['d'] ==5]

    build_plot(df,f'plot_3')

