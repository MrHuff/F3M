import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# from plots_KMVM_experiments import *
# ls_list = [1e-3,1e-2,1e-1,1,2,3,4,5,6,7,8,9,10,100,1000]
ls_list = [1e-3,1e-2,1e-1,1,2,3,'',5,'','','','',10,100,1000]
x=15
plt.rcParams['figure.figsize'] = 1.5*x, x

font_size = 60
plt.rcParams['font.size'] = font_size
plt.rcParams['legend.fontsize'] = 70
plt.rcParams['axes.labelsize'] = font_size
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = "serif"
# plt.rcParams['font.serif'] = "cm"
def error_plot_2(savefig, df, X, Y, slice,label_nice):
    mean = Y
    std = Y+' std'
    els = df[slice].unique().tolist()
    for label_df in els:
        sub_df = df[df[slice]==label_df]
        plt.scatter(np.log10(sub_df[X]), sub_df[mean].apply(lambda x: -30 if x==0 else np.log10(x)), marker='o',label=f'{label_nice}: {label_df}',alpha=1,s=100)
    # plt.tight_layout()
    plt.xticks(ticks= np.log10(df[X].unique()),labels=ls_list,rotation=0)
    plt.xlabel('Effective square distance $r$')
    plt.ylabel(r'Relative Error $\log_{10}(x)$')
    plt.legend()
    plt.savefig(savefig,bbox_inches = 'tight',pad_inches = 0)
    plt.clf()

def error_plot_3(savefig, df, X, Y, slice,label_nice):
    mean = Y
    std = Y+' std'
    els = df[slice].unique().tolist()
    for label_df in els:
        sub_df = df[df[slice]==label_df]
        plt.scatter(np.log10(sub_df[X]), sub_df[mean].apply(lambda x: -30 if x==0 else np.log10(x)), marker='o',label=f'{label_nice}: {label_df}',alpha=1,s=100)
    # plt.tight_layout()
    plt.xticks(ticks= np.log10(df[X].unique()),labels=ls_list,rotation=0)
    plt.xlabel('Effective square distance $r$')
    plt.ylabel(r'Abs error $\log_{10}(x)$')
    plt.legend()
    plt.savefig(savefig,bbox_inches = 'tight',pad_inches = 0)
    plt.clf()

def error_plot_4(savefig, df, X, Y,Y_2, slice,label_nice):
    mean = Y
    mean_2 = Y_2
    els = df[slice].unique().tolist()
    fig, ax1 = plt.subplots()
    ax1.set_xticks(ticks= np.log10(df[X].unique()))
    ax1.set_xticklabels(ls_list, )
    ax1.set_xlabel('Square distance $r$')
    ax1.set_ylabel(r'Relative Error $\log_{10}(x)$')
    ax2 = ax1.twinx()
    ax2.set_ylabel(r'Absolute error $\log_{10}(x)$')
    prop = ax1._get_lines.prop_cycler

    for label_df in els:
        color = next(prop)['color']
        sub_df = df[df[slice]==label_df]
        ax1.scatter(np.log10(sub_df[X]), sub_df[mean].apply(lambda x: -30 if x==0 else np.log10(x)), marker='o',alpha=0.6,s=700,color=color)
        ax2.scatter(np.log10(sub_df[X]), sub_df[mean_2].apply(lambda x: -30 if x==0 else np.log10(x)), marker='*',alpha=0.6,s=700,color=color)
        plt.plot([], [], 's' ,label=f'{label_nice}: {label_df}',color=color)

    # plt.tight_layout()
    legend = plt.legend(frameon=True,loc='upper right',framealpha=0.0)
    for legend_handle in legend.legendHandles:
        legend_handle._legmarker.set_markersize(25)
        # handle.set_sizes([100.0])

    legend_elements = [Line2D([0], [0],markerfacecolor='k',marker='o', color='w', label='Rel Error', markersize=40),
                          Line2D([0], [0],markerfacecolor='k', marker='*',color='w', label='Abs Error',markersize=40),
                       ]
    ax1.legend(handles=legend_elements)

    plt.savefig(savefig,bbox_inches = 'tight',pad_inches = 0)
    plt.clf()


if __name__ == '__main__':
    df = pd.read_csv("rbf_experiment_error_2.csv",index_col=0)
    for d in df['d'].unique().tolist():
        sub_df = df[df['d']==d]
        # error_plot_2(f"rbf_2_error_d={d}.png",sub_df,'eff_far_field','rel_error','nodes','nodes')
        # error_plot_3(f"rbf_2_error_d_abs={d}.png",sub_df,'eff_far_field','abs_error','nodes','nodes')
        error_plot_4(f"rbf_3_error_d_abs={d}.png",sub_df,'eff_far_field','rel_error','abs_error','nodes','r')





