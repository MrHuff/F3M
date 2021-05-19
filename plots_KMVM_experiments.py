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


x=8
plt.rcParams['figure.figsize'] = 1.5*x, x

font_size = 24
plt.rcParams['font.size'] = font_size
plt.rcParams['legend.fontsize'] = 24
plt.rcParams['axes.labelsize'] = font_size

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
fig, ax = plt.subplots()
prop = ax._get_lines.prop_cycler

def complexity_plots(savefig, df, X, Y, slice,label_nice):
    mean = Y
    els = df[slice].unique().tolist()

    big_X,big_Y = df[X],df[Y]


    for label_df in els:
        color = next(prop)['color']
        sub_df = df[df[slice]==label_df]
        m, b = np.polyfit(np.log10(sub_df[X]).astype(int), np.log10(sub_df[mean]), 1)
        print(m, b)
        plt.scatter(np.log10(sub_df[X]), np.log10(sub_df[mean]), marker='o',alpha=0.4,color=color)
        plt.plot(np.arange(min(np.log10(big_X)), max(np.log10(big_X)) + 1, step=1),
                 np.arange(min(np.log10(big_X)), max(np.log10(big_X)) + 1, step=1) * m + b,color=color)
        plt.plot([], [], '-o' ,label=f'{label_nice}: {label_df}',color=color)

    m, b = np.polyfit(np.log10(big_X), np.log10(big_Y), 1)
    print(m,b)
    plt.plot(np.arange(min(np.log10(big_X)), max(np.log10(big_X))+1, step=1),np.arange(min(np.log10(big_X)), max(np.log10(big_X))+1, step=1)*m+b,c='k',label=f'Complexity curve, slope={round(m,2)}', linewidth=4)
    plt.plot(np.arange(min(np.log10(big_X)), max(np.log10(big_X))+1, step=1),np.arange(min(np.log10(big_X)), max(np.log10(big_X))+1, step=1)*M_nlogn+b,c='k',linestyle='dashed',label='$\mathcal{O}(n\log n)$ curve, slope=1.11')
    plt.legend(prop={'size': 20},loc=0)
    plt.xticks(np.arange(min(np.log10(big_X)), max(np.log10(big_X))+1, step=1))

    plt.xlabel('$\log_{10}(N)$')
    plt.ylabel(r'$\log_{10}(T)$ seconds ')
    plt.savefig(savefig,bbox_inches = 'tight',pad_inches = 0)
    plt.clf()


def error_plot(savefig, df, X, Y, slice,label_nice):
    mean = Y
    std = Y+' std'
    els = df[slice].unique().tolist()
    for label_df in els:
        sub_df = df[df[slice]==label_df]
        plt.scatter(sub_df[X], sub_df[mean], marker='o',label=f'{label_nice}: {label_df}',alpha=0.4)
    plt.xlabel('Number of nodes $r$')
    plt.ylabel(r'Relative Error')
    plt.legend()
    plt.savefig(savefig,bbox_inches = 'tight',pad_inches = 0)
    plt.clf()


def get_worse_best(save_tex,df,X,Y):
    max = df.loc[df.groupby([X])[Y].idxmax()][['effective_variance','nr of node points',
            'effective variance limit','relative error 2', 'time (s)']]
    max.columns = ['EV','$r$','$\eta$','Relative Error','Time (s)']
    max.columns = pd.MultiIndex.from_product([['Worst'],max.columns ])
    min = df.loc[df.groupby([X])[Y].idxmin()][['n','effective_variance','nr of node points',
            'effective variance limit','relative error 2', 'time (s)']]
    min['n'] = np.log10(min['n']).astype(int)
    min.columns = [r'$\log_{10}(N)$','EV','$r$','$\eta$','Relative Error','Time (s)']

    min.columns = pd.MultiIndex.from_product([['Best'],min.columns ])
    max = max.reset_index(drop=True)
    min = min.reset_index(drop=True)
    df = pd.concat([min,max],axis=1)

    # df.loc[:,0]= np.log10(df['Best'][r'$\log_{10}(N)$'])
    df.to_latex(save_tex,escape=False,index=False)
    return df

def build_plot(df_name,plot_name):
    df = pd.read_csv(df_name,index_col=0)
    if not os.path.exists(plot_name):
        os.makedirs(plot_name)
    else:
        shutil.rmtree(plot_name)
        os.makedirs(plot_name)

    complexity_plots(f'{plot_name}/test_1.png', df, 'n', meanstd[1], 'effective_variance','EV')
    complexity_plots(f'{plot_name}/test_2.png', df, 'n', meanstd[1], 'effective variance limit','$\eta$')
    complexity_plots(f'{plot_name}/test_3.png', df, 'n', meanstd[1], 'nr of node points','$r$')
    error_plot(f'{plot_name}/test_4.png', df, 'nr of node points', meanstd[0], 'n','$N$') #Show independence of n
    error_plot(f'{plot_name}/test_5.png', df, 'nr of node points', meanstd[0], 'effective_variance','EV')
    error_plot(f'{plot_name}/test_6.png', df, 'nr of node points', meanstd[0], 'effective variance limit','$\eta$')
    get_worse_best(f'{plot_name}/test_1.tex',df,'n',meanstd[0])
    get_worse_best(f'{plot_name}/test_2.tex',df,'n',meanstd[1])

    doc = Document(default_filepath=f'{plot_name}/subfig_tex')
    string_append = ''
    with doc.create(Figure(position='H')) as plot:
        for i,p in enumerate(range(1,7)):
            p_str = f'{plot_name}/test_{p}.png'
            string_append+=r'\includegraphics[width=0.33\linewidth]{%s}'%p_str + '%\n'
            if (i+1)%3==0:
                with doc.create(subfigure(position='H', width=NoEscape(r'\linewidth'))):
                    doc.append(string_append)
                string_append=''
    doc.generate_tex()

if __name__ == '__main__':
    build_plot("experiment_1_results_summary.csv",'exp_1')
    # df = pd.read_csv("experiment_1_results_summary.csv",index_col=0)
    # complexity_plots('test_1.png', df, 'n', meanstd[1], 'effective_variance','Effective variance')
    # complexity_plots('test_2.png', df, 'n', meanstd[1], 'effective variance limit','$\eta$')
    # complexity_plots('test_3.png', df, 'n', meanstd[1], 'nr of node points','$r$')
    # error_plot('test_4.png', df, 'nr of node points', meanstd[0], 'n','$N$') #Show independence of n
    # error_plot('test_5.png', df, 'nr of node points', meanstd[0], 'effective_variance','Effective variance')
    # error_plot('test_6.png', df, 'nr of node points', meanstd[0], 'effective variance limit','$\eta$')
    # get_worse_best('test_1.tex',df,'n',meanstd[0])
    # get_worse_best('test_2.tex',df,'n',meanstd[1])

