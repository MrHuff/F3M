import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# from plots_KMVM_experiments import *
ls_list = [1e-3,1e-2,1e-1,1,2,3,4,5,6,7,8,9,10,100,1000]
x=15
plt.rcParams['figure.figsize'] = 1.5*x, x

font_size = 24
plt.rcParams['font.size'] = font_size
plt.rcParams['legend.fontsize'] = 24
plt.rcParams['axes.labelsize'] = font_size
def error_plot_2(savefig, df, X, Y, slice,label_nice):
    mean = Y
    std = Y+' std'
    els = df[slice].unique().tolist()
    for label_df in els:
        sub_df = df[df[slice]==label_df]
        plt.scatter(np.log10(sub_df[X]), sub_df[mean].apply(lambda x: -30 if x==0 else np.log10(x)), marker='o',label=f'{label_nice}: {label_df}',alpha=0.6)
    plt.tight_layout()
    plt.xticks(ticks= np.log10(df[X].unique()),labels=ls_list,rotation=90)
    plt.xlabel('Effective square distance $r$')
    plt.ylabel(r'Relative Error')
    plt.legend()
    plt.savefig(savefig,bbox_inches = 'tight',pad_inches = 0)
    plt.clf()

if __name__ == '__main__':
    df = pd.read_csv("rbf_experiment_error.csv",index_col=0)
    for d in df['d'].unique().tolist():
        sub_df = df[df['d']==d]
        error_plot_2(f"rbf_error_d={d}.png",sub_df,'eff_far_field','rel_error','nodes','nodes')
        error_plot_2(f"rbf_error_d_abs={d}.png",sub_df,'eff_far_field','abs_error','nodes','nodes')





