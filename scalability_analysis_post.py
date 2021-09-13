import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = "serif"
if __name__ == '__main__':
    cat_list = []
    for n in [1e6,1e7,1e8,1e9]:
        df = pd.read_csv(f'file_{int(n)}.csv')
        cat_list.append(df)

    subset = pd.concat(cat_list,ignore_index=True)
    subset = subset.rename(columns={"par_fac": "N GPU"}, errors="raise")
    s = '{10}'
    plot_table = subset.pivot(index='N GPU', columns='n', values='runtime').apply(lambda x :round(x,2))

    print(plot_table.to_markdown())

    for n in [1e6,1e7,1e8,1e9]:
        subset_2 = subset[subset['n']==n]

        plt.plot('N GPU','runtime',data=subset_2,linestyle='--', marker='o',label=fr'$\log_{s}(n)={int(np.log10(n))}$')
    plt.legend(prop={'size': 10})
    plt.xlabel('N GPU')
    plt.ylabel('Runtime')
    plt.savefig(f'scalability_plot.png',bbox_inches = 'tight',
pad_inches = 0.05)
    plt.clf()

